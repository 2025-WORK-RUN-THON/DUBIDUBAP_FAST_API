from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import select

from app.db import get_session, init_db
from app.models import Video, Transcript, AnalysisSummary
from app.services.ytdl import download_audio
from app.services.transcribe import Transcriber
from app.services.process import upsert_text_embedding_for_video
from app.core.config import settings
from app.utils.callback import post_with_retry
import threading


router = APIRouter(prefix="/process", tags=["process"])


class TranscribeFromUrlRequest(BaseModel):
    url: str = Field(..., description="YouTube watch URL")
    language: Optional[str] = None
    model_size: str = "small"
    dry_run: bool = False
    retries: int = 1


@router.post("/transcribe-url")
def transcribe_from_url(payload: TranscribeFromUrlRequest, session=Depends(get_session)):
    init_db()
    if "v=" not in payload.url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL (missing v=)")
    if not settings.ENABLE_TRANSCRIBE and not payload.dry_run:
        raise HTTPException(status_code=400, detail="Transcription disabled. Set ENABLE_TRANSCRIBE=true in .env")
    video_id_str = payload.url.split("v=")[-1].split("&")[0]

    video = session.exec(select(Video).where(Video.video_id == video_id_str)).first()
    if not video:
        video = Video(video_id=video_id_str, title=video_id_str, url=payload.url, view_count=0)
        session.add(video)
        session.flush()

    work_dir = os.path.join(os.getcwd(), "data", "audio")
    if payload.dry_run:
        return {
            "video_id": video.id,
            "planned_audio_path": os.path.join(work_dir, f"{video_id_str}.m4a"),
            "note": "dry_run=true → download/transcribe skipped",
        }

    try:
        audio_path = download_audio(payload.url, work_dir, filename_stem=video_id_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio download failed: {e}")

    # Transcription with simple fallback strategy
    attempt = 0
    last_err: Exception | None = None
    model_candidates = [payload.model_size or settings.TRANSCRIBE_MODEL_SIZE]
    if (payload.model_size or settings.TRANSCRIBE_MODEL_SIZE) != "tiny":
        model_candidates.append("tiny")
    for model_name in model_candidates:
        try:
            transcriber = Transcriber(model_size=model_name)
            text, lang, duration = transcriber.transcribe(audio_path, language=payload.language)
            break
        except Exception as e:
            last_err = e
            attempt += 1
            if attempt > payload.retries:
                continue
    else:
        raise HTTPException(status_code=502, detail=f"Transcription failed after retries: {last_err}")

    t = Transcript(video_id=video.id, text=text, language=lang, duration_sec=duration)
    session.add(t)
    session.flush()

    # Store text embedding for search
    upsert_text_embedding_for_video(video.id, text, kind="lyrics", session=session)

    result = {
        "video_id": video.id,
        "transcript_id": t.id,
        "language": lang,
        "duration_sec": duration,
        "text_preview": text[:200],
    }
    # Optional callback
    callback_url = settings.SPRING_CALLBACK_URL or None
    if callback_url:
        threading.Thread(target=post_with_retry, args=(callback_url, result), daemon=True).start()
    return result



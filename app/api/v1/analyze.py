from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import select

from app.db import get_session, init_db
from app.models import Video, Embedding, AnalysisSummary
from app.services.ytdl import download_audio
from app.services.audio_utils import load_mono_audio
from app.services.music import analyze_music
from app.services.pitch import compute_pitch_profile
from app.services.emotion import summarize_audio_emotion
from app.services.text_embed import embed_text
from app.services.lyrics import count_brand_mentions, detect_repeated_phrases
from app.services.embeddings import to_bytes


router = APIRouter(prefix="/analyze", tags=["analyze"])


class AnalyzeFromUrlRequest(BaseModel):
    url: str = Field(...)
    do_pitch: bool = True
    do_emotion: bool = True
    do_lyrics: bool = True
    brand_name: Optional[str] = None
    dry_run: bool = False


@router.post("/url")
def analyze_from_url(payload: AnalyzeFromUrlRequest, session=Depends(get_session)):
    init_db()
    if "v=" not in payload.url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL (missing v=)")
    video_id_str = payload.url.split("v=")[-1].split("&")[0]

    video = session.exec(select(Video).where(Video.video_id == video_id_str)).first()
    if not video:
        video = Video(video_id=video_id_str, title=video_id_str, url=payload.url, view_count=0)
        session.add(video)
        session.flush()

    work_dir = os.path.join(os.getcwd(), "data", "audio")
    if payload.dry_run:
        return {"video_id": video.id, "planned_audio_path": os.path.join(work_dir, f"{video_id_str}.m4a")}

    try:
        audio_path = download_audio(payload.url, work_dir, filename_stem=video_id_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio download failed: {e}")
    y, sr = load_mono_audio(audio_path, sr=16000)

    result = {"video_id": video.id}

    if payload.do_pitch:
        profile = compute_pitch_profile(y, sr)
        # Save as embedding-like bytes for simple storage (3-d vector)
        vec = to_bytes([profile["mean_hz"], profile["std_hz"], profile["median_hz"]])
        emb = Embedding(video_id=video.id, kind="pitch_profile", vector=vec, metadata_json="{}")
        session.add(emb)
        session.flush()
        session.add(AnalysisSummary(video_id=video.id, kind="pitch_profile", data_json=str(profile)))
        result["pitch_profile"] = profile

    # Always compute BPM/Key summary for reference
    try:
        music = analyze_music(y, sr)
        session.add(AnalysisSummary(video_id=video.id, kind="music_summary", data_json=str(music)))
        result["music_summary"] = music
    except Exception:
        pass

    if payload.do_emotion:
        emo = summarize_audio_emotion(y, sr)
        # Save top-5 probs as vector
        vec = to_bytes(list(emo.values()))
        emb = Embedding(video_id=video.id, kind="audio_emotion", vector=vec, metadata_json="{}")
        session.add(emb)
        session.flush()
        session.add(AnalysisSummary(video_id=video.id, kind="audio_emotion", data_json=str(emo)))
        result["emotion"] = emo

    # Optional lyrics-based analysis (requires transcript if available)
    if payload.do_lyrics:
        # Fetch latest transcript for this video if exists
        from app.models import Transcript
        t = session.exec(select(Transcript).where(Transcript.video_id == video.id).order_by(Transcript.id.desc())).first()
        if t and t.text:
            hook_candidates = detect_repeated_phrases(t.text)
            brand_count = count_brand_mentions(t.text, payload.brand_name or video.title)
            lyr_summary = {"hook_candidates": hook_candidates, "brand_mentions": int(brand_count)}
            session.add(AnalysisSummary(video_id=video.id, kind="lyrics_summary", data_json=str(lyr_summary)))
            result["lyrics_summary"] = lyr_summary

    return result



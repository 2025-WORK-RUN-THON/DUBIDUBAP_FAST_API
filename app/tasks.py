from __future__ import annotations

from celery import Celery

from app.core.config import settings


celery_app = Celery(
    "trendy_lyrics",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.task_always_eager = bool(settings.CELERY_ALWAYS_EAGER)


@celery_app.task
def add(x: int, y: int) -> int:
    return x + y


@celery_app.task
def transcribe_task(url: str, language: str | None = None) -> dict:
    from app.api.v1.process import Transcriber, download_audio, os, upsert_text_embedding_for_video, Transcript, Video, select
    from app.db import session_scope
    work_dir = os.path.join(os.getcwd(), "data", "audio")
    if "v=" not in url:
        raise ValueError("Invalid URL")
    video_id_str = url.split("v=")[-1].split("&")[0]
    with session_scope() as session:
        video = session.exec(select(Video).where(Video.video_id == video_id_str)).first()
        if not video:
            video = Video(video_id=video_id_str, title=video_id_str, url=url, view_count=0)
            session.add(video)
            session.flush()
        audio_path = download_audio(url, work_dir, filename_stem=video_id_str)
        transcriber = Transcriber(model_size=settings.TRANSCRIBE_MODEL_SIZE)
        text, lang, duration = transcriber.transcribe(audio_path, language=language)
        t = Transcript(video_id=video.id, text=text, language=lang, duration_sec=duration)
        session.add(t)
        session.flush()
        upsert_text_embedding_for_video(video.id, text, kind="lyrics", session=session)
        return {"video_id": video.id, "transcript_id": t.id, "language": lang, "duration_sec": duration}



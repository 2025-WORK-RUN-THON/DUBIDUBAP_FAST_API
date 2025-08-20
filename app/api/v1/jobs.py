from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.tasks import transcribe_task


router = APIRouter(prefix="/jobs", tags=["jobs"])


class TranscribeJobRequest(BaseModel):
    url: str
    language: str | None = None


@router.post("/transcribe")
def enqueue_transcription(payload: TranscribeJobRequest):
    task = transcribe_task.delay(payload.url, payload.language)
    return {"task_id": task.id, "eager": task.result is not None}



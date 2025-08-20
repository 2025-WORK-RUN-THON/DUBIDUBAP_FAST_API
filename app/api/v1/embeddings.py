from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import select

from app.db import get_session, init_db
from app.models import Embedding, Video
from app.services.embeddings import to_bytes
from app.services.process import upsert_text_embedding_for_video
from app.services.vector_store import upsert_vector


router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class UpsertEmbeddingRequest(BaseModel):
    video_id: int = Field(..., description="Video row id")
    kind: str = Field(default="lyrics")
    vector: list[float]
    metadata_json: Optional[str] = "{}"


@router.post("/upsert")
def upsert_embedding(payload: UpsertEmbeddingRequest, session=Depends(get_session)):
    init_db()
    video = session.exec(select(Video).where(Video.id == payload.video_id)).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    existing = session.exec(
        select(Embedding).where(Embedding.video_id == payload.video_id, Embedding.kind == payload.kind)
    ).first()

    vector_blob = to_bytes(payload.vector)

    if existing:
        existing.vector = vector_blob
        existing.metadata_json = payload.metadata_json or existing.metadata_json
        session.add(existing)
        session.flush()
        try:
            import numpy as np
            upsert_vector(session=session, kind=existing.kind, embedding_id=int(existing.id), video_id=int(existing.video_id), vector=np.frombuffer(existing.vector, dtype=np.float32))
        except Exception:
            pass
        return {"id": existing.id, "status": "updated"}

    emb = Embedding(
        video_id=payload.video_id,
        kind=payload.kind,
        vector=vector_blob,
        metadata_json=payload.metadata_json or "{}",
    )
    session.add(emb)
    session.flush()
    try:
        import numpy as np
        upsert_vector(session=session, kind=emb.kind, embedding_id=int(emb.id), video_id=int(emb.video_id), vector=np.frombuffer(emb.vector, dtype=np.float32))
    except Exception:
        pass
    return {"id": emb.id, "status": "created"}


class UpsertTextEmbeddingRequest(BaseModel):
    video_id: int
    text: str
    kind: str = "lyrics"


@router.post("/upsert-text")
def upsert_text_embedding(payload: UpsertTextEmbeddingRequest, session=Depends(get_session)):
    init_db()
    emb_id = upsert_text_embedding_for_video(
        video_id=payload.video_id, text=payload.text, kind=payload.kind, session=session
    )
    return {"id": emb_id, "status": "ok"}



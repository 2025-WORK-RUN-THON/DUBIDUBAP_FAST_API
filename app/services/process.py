from __future__ import annotations

import os
from typing import Optional

from sqlmodel import select, Session

from app.db import session_scope
from app.models import Video, Embedding
from app.services.text_embed import embed_text
from app.services.embeddings import to_bytes


def upsert_text_embedding_for_video(
    video_id: int,
    text: str,
    kind: str = "lyrics",
    session: Session | None = None,
) -> int:
    """Create or update a text embedding for a given video row id.

    Returns embedding row id.
    """
    vec = embed_text(text)
    blob = to_bytes(vec)

    if session is None:
        with session_scope() as scoped_session:
            video = scoped_session.exec(select(Video).where(Video.id == video_id)).first()
            if not video:
                raise ValueError("Video not found")

            existing = scoped_session.exec(
                select(Embedding).where(Embedding.video_id == video_id, Embedding.kind == kind)
            ).first()
            if existing:
                existing.vector = blob
                scoped_session.add(existing)
                scoped_session.flush()
                return int(existing.id)

            emb = Embedding(video_id=video_id, kind=kind, vector=blob, metadata_json="{}")
            scoped_session.add(emb)
            scoped_session.flush()
            return int(emb.id)
    else:
        video = session.exec(select(Video).where(Video.id == video_id)).first()
        if not video:
            raise ValueError("Video not found")

        existing = session.exec(
            select(Embedding).where(Embedding.video_id == video_id, Embedding.kind == kind)
        ).first()
        if existing:
            existing.vector = blob
            session.add(existing)
            session.flush()
            return int(existing.id)

        emb = Embedding(video_id=video_id, kind=kind, vector=blob, metadata_json="{}")
        session.add(emb)
        session.flush()
        return int(emb.id)



from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import select

from app.core.config import settings
from app.db import get_session, init_db
from app.models import Video, Embedding
from app.services.youtube import fetch_popular_ad_videos
from app.services.embeddings import to_bytes, from_bytes, knn_cosine
from app.services.process import upsert_text_embedding_for_video
from app.services.vector_store import search_vectors


router = APIRouter(prefix="/videos", tags=["videos"])


class FetchRequest(BaseModel):
    query: str = Field(..., examples=["'로고송' OR 'CM송' OR '광고음악' OR '브랜드송'"])
    min_view_count: int = 1_000_000
    target_count: int = 20
    api_key: Optional[str] = None
    ingest: bool = True


class VideoOut(BaseModel):
    id: int
    url: str
    title: str
    view_count: int


@router.post("/fetch", response_model=List[VideoOut])
def fetch_and_store_videos(payload: FetchRequest, session=Depends(get_session)):
    init_db()

    api_key = payload.api_key or settings.YOUTUBE_API_KEY or ""
    if not api_key:
        raise HTTPException(status_code=400, detail="YouTube API key is required (env YOUTUBE_API_KEY or request.api_key)")

    pairs = fetch_popular_ad_videos(
        api_key=api_key,
        query=payload.query,
        min_view_count=payload.min_view_count,
        target_count=payload.target_count,
    )

    created: list[Video] = []
    for url, title in pairs:
        video_id = url.split("v=")[-1]
        existing = session.exec(select(Video).where(Video.video_id == video_id)).first()
        if existing:
            continue
        video = Video(video_id=video_id, title=title, url=url, view_count=payload.min_view_count)
        session.add(video)
        session.flush()
        created.append(video)
        # 자동 임베딩(간단히 제목 기반) 또는 후속 배치 처리 준비
        if payload.ingest:
            try:
                upsert_text_embedding_for_video(video.id, title, kind="lyrics", session=session)
            except Exception:
                pass

    return [VideoOut(id=v.id, url=v.url, title=v.title, view_count=v.view_count) for v in created]


class SearchRequest(BaseModel):
    vector: list[float]
    kind: str = "lyrics"
    top_k: int = 5


@router.post("/search", response_model=List[VideoOut])
def search_by_vector(payload: SearchRequest, session=Depends(get_session)):
    init_db()
    embeddings = session.exec(select(Embedding).where(Embedding.kind == payload.kind)).all()
    if not embeddings:
        return []

    import numpy as np

    candidate_vecs = [from_bytes(e.vector) for e in embeddings]
    target_dim = int(candidate_vecs[0].shape[0])
    q = np.asarray(payload.vector, dtype=np.float32)
    # pad or truncate query to match stored embedding dimension
    if q.shape[0] < target_dim:
        pad = np.zeros(target_dim - q.shape[0], dtype=np.float32)
        query_vec = np.concatenate([q, pad], axis=0)
    elif q.shape[0] > target_dim:
        query_vec = q[:target_dim]
    else:
        query_vec = q
    # Try vector store first
    ordered_video_ids = search_vectors(session=session, kind=payload.kind, query=query_vec, top_k=payload.top_k)
    id_to_video: dict[int, Video] = {v.id: v for v in session.exec(select(Video)).all()}

    results: list[VideoOut] = []
    for vid in ordered_video_ids:
        video = id_to_video.get(vid)
        if video:
            results.append(VideoOut(id=video.id, url=video.url, title=video.title, view_count=video.view_count))
    return results


class IngestRequest(BaseModel):
    url: str
    title: str = ""
    view_count: int = 0
    initial_text: str | None = None


@router.post("/ingest", response_model=VideoOut)
def ingest_video(payload: IngestRequest, session=Depends(get_session)):
    init_db()
    if "v=" not in payload.url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL (missing v=)")
    video_id_str = payload.url.split("v=")[-1].split("&")[0]

    existing = session.exec(select(Video).where(Video.video_id == video_id_str)).first()
    if existing:
        video = existing
    else:
        video = Video(
            video_id=video_id_str,
            title=payload.title or video_id_str,
            url=payload.url,
            view_count=max(0, payload.view_count),
        )
        session.add(video)
        session.flush()

    if payload.initial_text:
        upsert_text_embedding_for_video(video.id, payload.initial_text, kind="lyrics", session=session)

    return VideoOut(id=video.id, url=video.url, title=video.title, view_count=video.view_count)



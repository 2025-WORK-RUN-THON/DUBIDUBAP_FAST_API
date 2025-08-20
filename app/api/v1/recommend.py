from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
import numpy as np
from sqlmodel import Session
import requests

from app.db import get_session, init_db
from app.services.recommend import recommend_videos
from pydantic import BaseModel


class RecommendFilters(BaseModel):
    min_view_count: int | None = None
    title_contains: str | None = None


router = APIRouter(prefix="/recommend", tags=["recommend"])


class RecommendRequest(BaseModel):
    top_k: int = 3
    kind_weights: Dict[str, float] = Field(
        default_factory=lambda: {"lyrics": 1.0, "audio_emotion": 0.5, "pitch_profile": 0.25, "music_summary": 0.25}
    )
    query_vectors: Dict[str, List[float]] = Field(
        default_factory=lambda: {"lyrics": [0.0] * 256}
    )
    callback_url: Optional[str] = None
    filters: Optional[RecommendFilters] = None


@router.post("")
def recommend(payload: RecommendRequest, session: Session = Depends(get_session)):
    init_db()
    qvecs = {k: np.asarray(v, dtype=np.float32) for k, v in payload.query_vectors.items()}
    pairs = recommend_videos(session, payload.kind_weights, qvecs, top_k=payload.top_k)
    # simple filtering after scoring
    if payload.filters:
        fv = []
        for v, score in pairs:
            if payload.filters.min_view_count is not None and v.view_count < payload.filters.min_view_count:
                continue
            if payload.filters.title_contains and payload.filters.title_contains not in v.title:
                continue
            fv.append((v, score))
        pairs = fv
    result = [
        {"id": v.id, "title": v.title, "url": v.url, "score": float(score)} for v, score in pairs
    ]
    if payload.callback_url:
        try:
            requests.post(payload.callback_url, json={"recommendations": result})
        except Exception:
            # Do not fail the request if callback fails
            pass
    return result



from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlmodel import select

from app.db import get_session, init_db
from app.models import Video, Transcript, Embedding, AnalysisSummary


router = APIRouter(prefix="/admin", tags=["admin"]) 


@router.get("/videos")
def list_videos(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0), session=Depends(get_session)):
    init_db()
    items = session.exec(select(Video).offset(offset).limit(limit)).all()
    return [
        {"id": v.id, "video_id": v.video_id, "title": v.title, "url": v.url, "view_count": v.view_count, "created_at": v.created_at}
        for v in items
    ]


@router.get("/transcripts")
def list_transcripts(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0), session=Depends(get_session)):
    init_db()
    items = session.exec(select(Transcript).offset(offset).limit(limit)).all()
    return [
        {"id": t.id, "video_id": t.video_id, "language": t.language, "duration_sec": t.duration_sec, "created_at": t.created_at, "text_len": len(t.text or "")}
        for t in items
    ]


@router.get("/embeddings")
def list_embeddings(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0), session=Depends(get_session)):
    init_db()
    items = session.exec(select(Embedding).offset(offset).limit(limit)).all()
    return [
        {"id": e.id, "video_id": e.video_id, "kind": e.kind, "created_at": e.created_at}
        for e in items
    ]


@router.get("/analysis")
def list_analysis(
    kind: str | None = Query(None, description="summary kind: pitch_profile|audio_emotion|lyrics_summary|music_summary"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session=Depends(get_session),
):
    init_db()
    stmt = select(AnalysisSummary)
    if kind:
        stmt = stmt.where(AnalysisSummary.kind == kind)
    items = session.exec(stmt.offset(offset).limit(limit)).all()
    return [
        {
            "id": a.id,
            "video_id": a.video_id,
            "kind": a.kind,
            "data": a.data_json,
            "created_at": a.created_at,
        }
        for a in items
    ]


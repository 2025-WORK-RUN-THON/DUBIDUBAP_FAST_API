from __future__ import annotations

from typing import List, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.trends import fetch_related_queries_cached


router = APIRouter(prefix="/trends", tags=["trends"])


class TrendsRequest(BaseModel):
    keywords: List[str] = Field(..., min_items=1)
    geo: str = "KR"
    dry_run: bool = False


@router.post("/related")
def related(payload: TrendsRequest) -> Dict[str, list[str]]:
    if payload.dry_run:
        return {kw: [f"{kw} 키워드1", f"{kw} 키워드2", f"{kw} 키워드3"] for kw in payload.keywords}
    try:
        return fetch_related_queries_cached(payload.keywords, geo=payload.geo)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Trends fetch failed: {e}")



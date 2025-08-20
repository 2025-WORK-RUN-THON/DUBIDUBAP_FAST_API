from __future__ import annotations

from typing import Dict, List, Optional
import uuid
from enum import Enum

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, model_validator
import requests

from app.db import get_session, init_db
from app.services.trends import fetch_related_queries
from app.services.recommend import recommend_videos
from app.services.prompt import build_master_prompt, build_suno_prompt_ko
from app.core.config import settings
from app.services.text_embed import embed_text
from app.services.embeddings import to_bytes
from app.models import Embedding, Video, Prompt, AnalysisSummary
from sqlmodel import select
import numpy as np
import threading
from app.utils.callback import post_with_retry
import json
from ast import literal_eval


router = APIRouter(prefix="/generate", tags=["generate"])


class VersionType(str, Enum):
    SHORT = "SHORT"
    LONG = "LONG"


class SunoModel(str, Enum):
    V3_5 = "V3_5"
    V4 = "V4"
    V4_5 = "V4_5"


class VocalGender(str, Enum):
    m = "m"
    f = "f"


class GenerateRequest(BaseModel):
    service_name: str = Field(min_length=1, max_length=80)
    slogan: str
    target_customer: str
    mood_tone: List[str]
    music_genre: str = Field(min_length=1, max_length=200)
    version: VersionType
    industry: Optional[str] = None
    marketing_item: Optional[str] = None
    extra: Optional[str] = None
    trend_keywords: List[str] = Field(default_factory=list)
    top_k_examples: int = 3
    callback_url: Optional[str] = None
    request_id: Optional[str] = None
    generate_in: str = Field(default="fastapi", pattern="^(fastapi|spring)$")
    # Suno 전달용(스프링이 Suno API에 쓸 옵션들)
    suno_custom_mode: Optional[bool] = None
    suno_instrumental: Optional[bool] = None
    suno_model: Optional[SunoModel] = None  # V3_5|V4|V4_5
    suno_negative_tags: Optional[str] = None
    suno_vocal_gender: Optional[VocalGender] = None  # m|f
    suno_style_weight: Optional[float] = None
    suno_weirdness: Optional[float] = None
    suno_audio_weight: Optional[float] = None
    # 편의 플래그(베이스로 복사)
    instrumental: Optional[bool] = None
    vocal_gender: Optional[VocalGender] = None  # m|f

    @model_validator(mode="after")
    def clamp_weights(self):
        def _clamp_round(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None
            try:
                y = max(0.0, min(1.0, float(x)))
                return round(y, 2)
            except Exception:
                return None

        self.suno_style_weight = _clamp_round(self.suno_style_weight)
        self.suno_weirdness = _clamp_round(self.suno_weirdness)
        self.suno_audio_weight = _clamp_round(self.suno_audio_weight)
        return self


@router.post("")
def generate(payload: GenerateRequest, session=Depends(get_session)):
    init_db()
    req_id = payload.request_id or str(uuid.uuid4())
    trends = []
    if payload.trend_keywords:
        try:
            related = fetch_related_queries(payload.trend_keywords)
            for vals in related.values():
                trends.extend(vals[:3])
        except Exception:
            trends = payload.trend_keywords

    # Recommend examples using simple lyrics embedding of service name + slogan
    qvec = embed_text(f"{payload.service_name} {payload.slogan}")
    pairs = recommend_videos(
        session,
        kind_weights={"lyrics": 1.0},
        query_vectors={"lyrics": qvec},
        top_k=payload.top_k_examples,
    )
    example_videos = [v for v, _ in pairs]
    examples = [{"title": v.title, "url": v.url} for v in example_videos]

    # Gather analysis summaries for example videos to provide hints
    def _parse_jsonish(s: str):
        try:
            return json.loads(s)
        except Exception:
            try:
                return literal_eval(s)
            except Exception:
                return None

    music_items, emo_items, hooks_items = [], [], []
    if example_videos:
        vid_ids = [v.id for v in example_videos]
        summaries = session.exec(
            select(AnalysisSummary).where(AnalysisSummary.video_id.in_(vid_ids))
        ).all()
        for s in summaries:
            parsed = _parse_jsonish(s.data_json or "{}") or {}
            if s.kind == "music_summary":
                music_items.append(parsed)
            elif s.kind == "audio_emotion":
                emo_items.append(parsed)
            elif s.kind == "lyrics_summary":
                hooks = parsed.get("hook_candidates") or []
                hooks_items.extend(hooks)

    # Aggregate music summary
    music_summary = None
    if music_items:
        try:
            import statistics as st
            bpms = [float(x.get("bpm", 0.0)) for x in music_items if x.get("bpm") is not None]
            mean_bpm = float(st.mean(bpms)) if bpms else 0.0
            keys = [x.get("key") for x in music_items if x.get("key")]
            modes = [x.get("mode") for x in music_items if x.get("mode")]
            key = max(set(keys), key=keys.count) if keys else None
            mode = max(set(modes), key=modes.count) if modes else None
            music_summary = {"bpm": round(mean_bpm, 2), "key": key, "mode": mode}
        except Exception:
            music_summary = None

    # Aggregate emotion
    emotion_hint = None
    if emo_items:
        from collections import defaultdict
        acc, n = defaultdict(float), 0
        for d in emo_items:
            for k, v in d.items():
                try:
                    acc[k] += float(v)
                except Exception:
                    pass
            n += 1
        if n > 0:
            emotion_hint = {k: round(v / n, 4) for k, v in acc.items()}

    # Hooks: pick top frequent phrases
    hooks = []
    if hooks_items:
        try:
            norm = []
            for item in hooks_items:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    phrase, cnt = item[0], item[1]
                    norm.append((str(phrase), int(cnt)))
            norm.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            hooks = [p for p, _ in norm[:5]]
        except Exception:
            hooks = []

    # 추가 정보 결합
    extra_combined = payload.extra or ""
    if payload.industry:
        extra_combined += (" " if extra_combined else "") + f"업종: {payload.industry}"
    if payload.marketing_item:
        extra_combined += (" " if extra_combined else "") + f"마케팅 포인트: {payload.marketing_item}"

    master_prompt = build_master_prompt(
        payload.service_name,
        payload.slogan,
        payload.target_customer,
        payload.mood_tone,
        payload.music_genre,
        payload.version,
        trends,
        examples,
        extra_combined,
    )
    suno_prompt = build_suno_prompt_ko(
        payload.service_name,
        payload.mood_tone,
        payload.music_genre,
        payload.version,
    )

    # Suno 콜백용 요청 구성: 스프링 서버가 Suno API에 그대로 전달 가능하도록 빌드
    _custom = payload.suno_custom_mode if payload.suno_custom_mode is not None else settings.SUNO_DEFAULT_CUSTOM_MODE
    _model = payload.suno_model or settings.SUNO_MODEL
    _cb = settings.SPRING_SUNO_CALLBACK_URL or settings.SPRING_CALLBACK_URL or ""
    if not _custom:
        # 정책상 customMode=true 고정이나, 안전하게 분기 유지
        suno_request = {
            "customMode": False,
            "model": _model,
            "callBackUrl": _cb,
            "prompt": master_prompt[:400],
        }
    else:
        # Custom Mode: 규칙에 따라 필드 포함. 스타일/타이틀 길이 제한 적용
        # V3_5/V4: style<=200, prompt<=3000. V4_5: style<=1000, prompt<=5000
        style_limit = 200 if _model in ("V3_5", "V4") else 1000
        prompt_limit = 3000 if _model in ("V3_5", "V4") else 5000
        # 스프링이 GPT-5-mini Responses API로 가사 생성 후 prompt를 채우므로 여기서는 비워 전달
        suno_request = {
            "customMode": True,
            "instrumental": (
                payload.suno_instrumental
                if payload.suno_instrumental is not None
                else (payload.instrumental if payload.instrumental is not None else settings.SUNO_DEFAULT_INSTRUMENTAL)
            ),
            "model": _model,
            "callBackUrl": _cb,
            "prompt": "",
            "style": payload.music_genre[:style_limit],
            "title": payload.service_name[:80],
            "negativeTags": (payload.suno_negative_tags if payload.suno_negative_tags is not None else settings.SUNO_NEGATIVE_TAGS)[:style_limit],
            "vocalGender": (
                payload.suno_vocal_gender
                if payload.suno_vocal_gender is not None
                else (payload.vocal_gender if payload.vocal_gender is not None else settings.SUNO_DEFAULT_VOCAL_GENDER)
            ) or None,
            "styleWeight": round(float(payload.suno_style_weight if payload.suno_style_weight is not None else settings.SUNO_DEFAULT_STYLE_WEIGHT), 2),
            "weirdnessConstraint": round(float(payload.suno_weirdness if payload.suno_weirdness is not None else settings.SUNO_DEFAULT_WEIRDNESS), 2),
            "audioWeight": round(float(payload.suno_audio_weight if payload.suno_audio_weight is not None else settings.SUNO_DEFAULT_AUDIO_WEIGHT), 2),
        }

    analysis = {
        "trends": trends,
        "examples": examples,
        "musicSummary": music_summary,
        "emotionHint": emotion_hint,
        "hooks": hooks,
    }

    result = {
        "master_prompt": master_prompt,
        "suno_prompt": suno_prompt,
        "trends": trends,
        "examples": examples,
        "analysis": analysis,
        "suno_request": suno_request,
        "sunoRequestBase": suno_request,
        "requestId": req_id,
    }

    # Decide where to generate: fastapi builds prompts already; for "spring", we forward inputs only
    if payload.generate_in == "spring":
        spring_payload = {
            "requestId": req_id,
            "inputs": {
                "serviceName": payload.service_name,
                "slogan": payload.slogan,
                "targetCustomer": payload.target_customer,
                "moodTone": payload.mood_tone,
                "musicGenre": payload.music_genre,
                "version": payload.version,
                "extra": payload.extra,
                "masterPrompt": master_prompt,
                "analysis": analysis,
            },
            "trends": trends,
            "examples": examples,
            "sunoRequestBase": suno_request,
        }
        if payload.callback_url:
            threading.Thread(target=post_with_retry, args=(payload.callback_url, spring_payload), daemon=True).start()
        return {"status": "forwarded_to_spring", "requestId": req_id}

    # fastapi path: persist prompt, respond immediately; optional callback in background
    try:
        from json import dumps
        p = Prompt(master_prompt=master_prompt, suno_prompt=suno_prompt, payload_json=dumps(payload.model_dump()))
        session.add(p)
        session.flush()
    except Exception:
        pass

    # optional callback
    if payload.callback_url:
        threading.Thread(target=post_with_retry, args=(payload.callback_url, result), daemon=True).start()
    return result



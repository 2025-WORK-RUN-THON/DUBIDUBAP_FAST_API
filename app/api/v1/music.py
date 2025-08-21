"""
음악 분석 API 엔드포인트
기존 generate 엔드포인트 개선 및 새로운 기능 추가
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel, Field

from app.services.music_analysis import music_analysis_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["music"])


# Request/Response 모델
class GenerateRequest(BaseModel):
    """기존 generate 요청 모델 (Spring Boot 호환)"""
    service_name: str
    slogan: Optional[str] = ""
    target_customer: Optional[str] = ""
    mood_tone: List[str] = Field(default_factory=list)
    music_genre: Optional[str] = "pop"
    version: str = "V1"
    industry: Optional[str] = ""
    marketing_item: Optional[str] = ""
    extra: Optional[str] = ""
    generate_in: str = "spring"
    request_id: str


class MusicSummary(BaseModel):
    """음악 요약 정보"""
    bpm: float
    key: str
    mode: str
    genre: str = "pop"


class Analysis(BaseModel):
    """분석 결과"""
    music_summary: MusicSummary


class SunoRequestBase(BaseModel):
    """Suno 요청 기본 정보"""
    style_weight: float = 0.7
    weirdness_constraint: float = 0.3
    audio_weight: float = 0.6


class GenerateResponse(BaseModel):
    """기존 generate 응답 모델 (Spring Boot 호환)"""
    request_id: str
    master_prompt: str
    examples: List[Dict[str, Any]]
    analysis: Analysis
    suno_request_base: SunoRequestBase


class MusicSearchRequest(BaseModel):
    """음악 검색 요청"""
    query: str
    type: str = "combined"
    limit: int = 10
    threshold: float = 0.3


class MusicSearchResponse(BaseModel):
    """음악 검색 응답"""
    request_id: str
    query: str
    results: List[Dict[str, Any]]
    total_count: int
    processing_time_ms: int


@router.post("/generate", response_model=GenerateResponse)
async def generate_with_analysis(
    request: GenerateRequest,
    x_request_id: Optional[str] = Header(None)
):
    """
    기존 Spring Boot 호환 generate 엔드포인트 (개선됨)
    실제 음악 분석 데이터를 활용하여 의미있는 응답 제공
    """
    request_id = x_request_id or request.request_id or str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        f"Generate 요청: service_name={request.service_name}, "
        f"genre={request.music_genre}, requestId={request_id}"
    )
    
    try:
        # 1. 서비스명 기반 유사 음악 검색
        similar_videos = await music_analysis_service.find_similar_music(
            query=f"{request.service_name} {request.slogan}".strip(),
            search_type="combined",
            limit=5,
            threshold=0.2
        )
        
        # 2. 음악 특성 분석
        music_summary = await music_analysis_service.generate_music_summary(similar_videos)
        
        # 3. 마스터 프롬프트 생성
        master_prompt = generate_master_prompt(request, similar_videos)
        
        # 4. 응답 구성 (기존 Spring Boot 형식 유지)
        response = GenerateResponse(
            request_id=request_id,
            master_prompt=master_prompt,
            examples=[
                {
                    "title": video.get("title", ""),
                    "views": video.get("views", 0),
                    "similarity_score": video.get("similarity_score", 0.0),
                    "video_id": video.get("video_id", "")
                }
                for video in similar_videos[:3]
            ],
            analysis=Analysis(
                music_summary=MusicSummary(
                    bpm=music_summary["bpm"],
                    key=music_summary["key"],
                    mode=music_summary["mode"],
                    genre=music_summary["genre"]
                )
            ),
            suno_request_base=SunoRequestBase(
                style_weight=0.65,
                weirdness_constraint=0.35,
                audio_weight=0.60
            )
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Generate 응답 완료: requestId={request_id}, "
            f"examples={len(response.examples)}, "
            f"bmp={music_summary['bpm']}, "
            f"processingTime={processing_time}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Generate 처리 중 오류: requestId={request_id}, error={e}")
        
        # 기본 응답 반환 (기존 동작 유지)
        return GenerateResponse(
            request_id=request_id,
            master_prompt=f"Generated prompt for {request.service_name}",
            examples=[],
            analysis=Analysis(
                music_summary=MusicSummary(
                    bpm=120.0,
                    key="C",
                    mode="major",
                    genre=request.music_genre or "pop"
                )
            ),
            suno_request_base=SunoRequestBase()
        )


@router.post("/music/search", response_model=MusicSearchResponse)
async def search_music(
    request: MusicSearchRequest,
    x_request_id: Optional[str] = Header(None)
):
    """음악 검색 API"""
    request_id = x_request_id or str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"음악 검색: query={request.query}, requestId={request_id}")
    
    try:
        results = await music_analysis_service.find_similar_music(
            query=request.query,
            search_type=request.type,
            limit=request.limit,
            threshold=request.threshold
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        response = MusicSearchResponse(
            request_id=request_id,
            query=request.query,
            results=results,
            total_count=len(results),
            processing_time_ms=processing_time
        )
        
        logger.info(f"음악 검색 완료: {len(results)}개 결과, {processing_time}ms")
        return response
        
    except Exception as e:
        logger.error(f"음악 검색 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/music/recommend/{video_id}")
async def get_music_recommendations(
    video_id: str,
    limit: int = Query(5, ge=1, le=20),
    x_request_id: Optional[str] = Header(None)
):
    """음악 추천 API"""
    request_id = x_request_id or str(uuid.uuid4())
    
    logger.info(f"음악 추천: video_id={video_id}, requestId={request_id}")
    
    try:
        recommendations = await music_analysis_service.get_music_recommendations(
            video_id=video_id,
            limit=limit
        )
        
        return {
            "request_id": request_id,
            "video_id": video_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"음악 추천 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/analysis")
async def get_trends_analysis(
    period: str = Query("all", description="분석 기간"),
    x_request_id: Optional[str] = Header(None)
):
    """트렌드 분석 API"""
    request_id = x_request_id or str(uuid.uuid4())
    
    logger.info(f"트렌드 분석: period={period}, requestId={request_id}")
    
    try:
        trends = await music_analysis_service.analyze_music_trends(period=period)
        trends["request_id"] = request_id
        
        return trends
        
    except Exception as e:
        logger.error(f"트렌드 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embeddings/{video_id}")
async def get_video_embeddings(
    video_id: str,
    include_embedding: bool = Query(False, description="임베딩 벡터 포함 여부")
):
    """특정 비디오 임베딩 정보 조회"""
    try:
        if not music_analysis_service.data_loaded:
            raise HTTPException(status_code=503, detail="임베딩 데이터가 로드되지 않았습니다")
        
        # 해당 비디오 찾기
        target_video = None
        for video in music_analysis_service.embeddings_data:
            if video['video_id'] == video_id:
                target_video = video.copy()
                break
        
        if target_video is None:
            raise HTTPException(status_code=404, detail="비디오를 찾을 수 없습니다")
        
        # 임베딩 벡터 제어
        if not include_embedding:
            target_video.pop('embedding', None)
        
        return {
            "video_id": video_id,
            "data": target_video
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"임베딩 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {
        "status": "healthy",
        "service": "Trendy Lyrics Music Analysis",
        "data_loaded": music_analysis_service.data_loaded,
        "video_count": len(music_analysis_service.embeddings_data) if music_analysis_service.data_loaded else 0
    }


def generate_master_prompt(request: GenerateRequest, similar_videos: List[Dict]) -> str:
    """마스터 프롬프트 생성"""
    
    if not similar_videos:
        return f"Create a logo song for {request.service_name} in {request.music_genre} style."
    
    # 유사 비디오 기반 특성 추출
    popular_keywords = []
    for video in similar_videos:
        title = video.get('title', '')
        words = title.split()
        popular_keywords.extend([w for w in words if len(w) > 1])
    
    # 빈도 기반 상위 키워드
    keyword_freq = {}
    for keyword in popular_keywords:
        keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
    
    top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:3]
    keywords_str = ", ".join([kw[0] for kw in top_keywords])
    
    prompt_parts = [
        f"Create a catchy logo song for '{request.service_name}'",
        f"Genre: {request.music_genre}",
        f"Target: {request.target_customer}" if request.target_customer else "",
        f"Slogan: {request.slogan}" if request.slogan else "",
        f"Trending elements: {keywords_str}" if keywords_str else "",
        f"Similar successful songs show high engagement with: {', '.join([v['title'][:30] for v in similar_videos[:2]])}"
    ]
    
    return " | ".join([part for part in prompt_parts if part])
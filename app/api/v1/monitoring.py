"""
모니터링 및 메트릭 API 엔드포인트
시스템 상태, 성능 지표, 헬스체크 제공
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, Optional, List
import time
import psutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.core.logging import logger, LogCategory
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="기본 헬스체크")
async def health_check():
    """기본 헬스체크 - 서비스 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.VERSION,
        "environment": settings.ENV
    }


@router.get("/health/detailed", summary="상세 헬스체크")
async def detailed_health_check():
    """상세 헬스체크 - 시스템 리소스 및 서비스 상태"""
    
    try:
        # 시스템 리소스 확인
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 프로세스 정보
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # 임베딩 데이터 확인
        embeddings_file = Path("data/processed/processed_embeddings_final.json")
        embeddings_status = embeddings_file.exists()
        embeddings_size = embeddings_file.stat().st_size if embeddings_status else 0
        
        # 모델 캐시 상태 확인
        from app.services.model_cache import model_cache
        cache_stats = model_cache.get_cache_stats()
        
        # 시스템 상태 판정
        status = "healthy"
        issues = []
        
        if memory.percent > 90:
            status = "degraded"
            issues.append("High memory usage")
        
        if disk.percent > 90:
            status = "degraded"
            issues.append("High disk usage")
        
        if cpu_percent > 95:
            status = "degraded"
            issues.append("High CPU usage")
        
        if not embeddings_status:
            status = "degraded" 
            issues.append("Embeddings data not found")
        
        health_data = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": settings.VERSION,
            "environment": settings.ENV,
            "uptime_seconds": time.time() - psutil.boot_time(),
            "issues": issues,
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_mb": round(memory.total / 1024 / 1024, 2),
                    "used_mb": round(memory.used / 1024 / 1024, 2),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / 1024 / 1024 / 1024, 2),
                    "used_gb": round(disk.used / 1024 / 1024 / 1024, 2),
                    "percent": disk.percent
                }
            },
            "process": {
                "memory_mb": round(process_memory.rss / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            },
            "services": {
                "embeddings_loaded": embeddings_status,
                "embeddings_size_mb": round(embeddings_size / 1024 / 1024, 2) if embeddings_size else 0,
                "model_cache": cache_stats,
                "python_version": sys.version.split()[0]
            }
        }
        
        # 로그 기록
        logger.info(
            f"Health check completed - Status: {status}",
            category=LogCategory.SYSTEM,
            health_status=status,
            issues_count=len(issues)
        )
        
        return health_data
        
    except Exception as e:
        logger.error(
            f"Health check failed: {str(e)}",
            category=LogCategory.SYSTEM
        )
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }


@router.get("/metrics", summary="성능 메트릭")
async def get_metrics(
    category: Optional[str] = Query(None, description="메트릭 카테고리 필터"),
    minutes: int = Query(60, description="최근 N분간의 메트릭", ge=1, le=1440)
):
    """성능 메트릭 조회"""
    
    try:
        # 메트릭 요약 가져오기
        metrics_summary = logger.get_metrics_summary(category=category, minutes=minutes)
        
        # 현재 시스템 메트릭 추가
        current_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        response = {
            "current": current_metrics,
            "summary": metrics_summary,
            "collection_period_minutes": minutes
        }
        
        logger.info(
            f"Metrics retrieved - Category: {category}, Period: {minutes}min",
            category=LogCategory.PERFORMANCE,
            metrics_count=metrics_summary.get("total_requests", 0)
        )
        
        return response
        
    except Exception as e:
        logger.error(
            f"Failed to retrieve metrics: {str(e)}",
            category=LogCategory.PERFORMANCE
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@router.get("/metrics/categories", summary="메트릭 카테고리 목록")
async def get_metric_categories():
    """사용 가능한 메트릭 카테고리 목록"""
    
    categories = [
        {
            "name": "system",
            "description": "시스템 전반적인 메트릭"
        },
        {
            "name": "api", 
            "description": "API 호출 관련 메트릭"
        },
        {
            "name": "youtube",
            "description": "YouTube 처리 관련 메트릭"
        },
        {
            "name": "transcription",
            "description": "음성 전사 관련 메트릭"
        },
        {
            "name": "embedding",
            "description": "임베딩 생성 관련 메트릭"
        },
        {
            "name": "music_analysis",
            "description": "음악 분석 관련 메트릭"
        },
        {
            "name": "emotion_analysis",
            "description": "감정 분석 관련 메트릭"
        },
        {
            "name": "model_cache",
            "description": "모델 캐시 관련 메트릭"
        },
        {
            "name": "performance",
            "description": "성능 관련 메트릭"
        }
    ]
    
    return {
        "categories": categories,
        "total": len(categories)
    }


@router.get("/logs/recent", summary="최근 로그")
async def get_recent_logs(
    level: Optional[str] = Query(None, description="로그 레벨 필터 (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    category: Optional[str] = Query(None, description="로그 카테고리 필터"),
    lines: int = Query(100, description="반환할 로그 라인 수", ge=1, le=1000)
):
    """최근 로그 조회 (파일 기반)"""
    
    try:
        if not settings.LOG_FILE:
            raise HTTPException(status_code=404, detail="Log file not configured")
        
        log_file = Path(settings.LOG_FILE)
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Log file not found")
        
        # 파일에서 최근 로그 읽기
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        # 최근 N개 라인 가져오기
        recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        # JSON 파싱 시도
        parsed_logs = []
        for line in recent_lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                import json
                log_entry = json.loads(line)
                
                # 필터 적용
                if level and log_entry.get('level') != level.upper():
                    continue
                
                if category and log_entry.get('category') != category:
                    continue
                
                parsed_logs.append(log_entry)
                
            except json.JSONDecodeError:
                # JSON 파싱 실패시 원문 그대로 포함
                parsed_logs.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "UNKNOWN",
                    "message": line,
                    "parsed": False
                })
        
        logger.info(
            f"Recent logs retrieved - Lines: {len(parsed_logs)}, Level: {level}, Category: {category}",
            category=LogCategory.SYSTEM
        )
        
        return {
            "logs": parsed_logs,
            "total_lines": len(parsed_logs),
            "filters": {
                "level": level,
                "category": category,
                "max_lines": lines
            }
        }
        
    except Exception as e:
        logger.error(
            f"Failed to retrieve logs: {str(e)}",
            category=LogCategory.SYSTEM
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")


@router.get("/status/services", summary="서비스 상태")
async def get_service_status():
    """각 서비스 컴포넌트 상태 확인"""
    
    try:
        services = {}
        
        # 1. 임베딩 서비스
        try:
            from app.services.music_analysis import music_analysis_service
            embeddings_loaded = hasattr(music_analysis_service, 'embeddings_data') and \
                              music_analysis_service.embeddings_data is not None
            services["music_analysis"] = {
                "status": "healthy" if embeddings_loaded else "degraded",
                "embeddings_loaded": embeddings_loaded,
                "data_count": len(music_analysis_service.embeddings_data) if embeddings_loaded else 0
            }
        except Exception as e:
            services["music_analysis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # 2. 감정 분석 서비스
        try:
            from app.services.emotion import _get_emotion_pipeline
            pipeline = _get_emotion_pipeline()
            services["emotion_analysis"] = {
                "status": "healthy" if pipeline else "degraded",
                "available": pipeline is not None,
                "device": "cuda" if pipeline and hasattr(pipeline, 'device') else "cpu"
            }
        except Exception as e:
            services["emotion_analysis"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
        
        # 3. 모델 캐시
        try:
            from app.services.model_cache import model_cache
            cache_stats = model_cache.get_cache_stats()
            services["model_cache"] = {
                "status": "healthy",
                "loaded_models": len(cache_stats.get("loaded_models", [])),
                "memory_usage_mb": cache_stats.get("memory_usage_mb", 0),
                "memory_usage_percent": cache_stats.get("memory_usage_percent", 0)
            }
        except Exception as e:
            services["model_cache"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # 전체 상태 판정
        overall_status = "healthy"
        unhealthy_count = sum(1 for s in services.values() if s.get("status") == "unhealthy")
        degraded_count = sum(1 for s in services.values() if s.get("status") == "degraded")
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        
        response = {
            "overall_status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": services,
            "summary": {
                "total_services": len(services),
                "healthy": sum(1 for s in services.values() if s.get("status") == "healthy"),
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            }
        }
        
        logger.info(
            f"Service status check completed - Overall: {overall_status}",
            category=LogCategory.SYSTEM,
            healthy_services=response["summary"]["healthy"],
            total_services=len(services)
        )
        
        return response
        
    except Exception as e:
        logger.error(
            f"Service status check failed: {str(e)}",
            category=LogCategory.SYSTEM
        )
        raise HTTPException(status_code=500, detail="Service status check failed")
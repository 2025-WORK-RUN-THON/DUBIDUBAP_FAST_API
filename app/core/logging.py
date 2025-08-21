"""
고급 로깅 및 모니터링 시스템
구조화된 로그, 성능 메트릭, 알림 기능
"""

import logging
import logging.handlers
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import traceback
from functools import wraps
import threading
from dataclasses import dataclass, asdict
from enum import Enum

from app.core.config import settings


class LogLevel(Enum):
    """로그 레벨 정의"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """로그 카테고리"""
    SYSTEM = "system"
    API = "api"
    DATABASE = "database"
    YOUTUBE = "youtube"
    TRANSCRIPTION = "transcription"
    EMBEDDING = "embedding"
    MUSIC_ANALYSIS = "music_analysis"
    EMOTION_ANALYSIS = "emotion_analysis"
    MODEL_CACHE = "model_cache"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ERROR = "error"


@dataclass
class LogMetrics:
    """성능 메트릭 데이터"""
    timestamp: str
    request_id: str
    category: str
    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class StructuredFormatter(logging.Formatter):
    """구조화된 JSON 로그 포맷터"""
    
    def format(self, record: logging.LogRecord) -> str:
        # 기본 로그 데이터
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # 추가 컨텍스트 정보
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, 'category'):
            log_data["category"] = record.category
        
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        
        if hasattr(record, 'performance'):
            log_data["performance"] = record.performance
        
        # 예외 정보
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # 추가 데이터
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data, ensure_ascii=False, separators=(',', ':'))


class TrendyLyricsLogger:
    """Trendy Lyrics 전용 로거"""
    
    def __init__(self, name: str = "trendy_lyrics"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.metrics_store = []
        self.metrics_lock = threading.Lock()
        self._setup_logger()
    
    def _setup_logger(self):
        """로거 초기 설정"""
        self.logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러 (로테이션)
        if settings.LOG_FILE:
            log_file = Path(settings.LOG_FILE)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self._parse_size(getattr(settings, 'LOG_MAX_SIZE', '10MB')),
                backupCount=getattr(settings, 'LOG_BACKUP_COUNT', 5),
                encoding='utf-8'
            )
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)
        
        # 에러 전용 파일 핸들러
        if settings.LOG_FILE:
            error_file = log_file.parent / f"error_{log_file.name}"
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=self._parse_size(getattr(settings, 'LOG_MAX_SIZE', '10MB')),
                backupCount=getattr(settings, 'LOG_BACKUP_COUNT', 5),
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(error_handler)
    
    def _parse_size(self, size_str: str) -> int:
        """크기 문자열을 바이트로 변환"""
        size_str = size_str.upper()
        if size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def log(self, level: LogLevel, message: str, category: LogCategory = LogCategory.SYSTEM, 
            request_id: Optional[str] = None, **kwargs):
        """구조화된 로그 기록"""
        extra = {
            'category': category.value,
            'request_id': request_id,
            'extra_data': kwargs
        }
        
        log_method = getattr(self.logger, level.value.lower())
        log_method(message, extra=extra)
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, 
             request_id: Optional[str] = None, **kwargs):
        self.log(LogLevel.INFO, message, category, request_id, **kwargs)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM,
                request_id: Optional[str] = None, **kwargs):
        self.log(LogLevel.WARNING, message, category, request_id, **kwargs)
    
    def error(self, message: str, category: LogCategory = LogCategory.ERROR,
              request_id: Optional[str] = None, exc_info: bool = True, **kwargs):
        extra = {
            'category': category.value,
            'request_id': request_id,
            'extra_data': kwargs
        }
        self.logger.error(message, extra=extra, exc_info=exc_info)
    
    def critical(self, message: str, category: LogCategory = LogCategory.ERROR,
                 request_id: Optional[str] = None, **kwargs):
        self.log(LogLevel.CRITICAL, message, category, request_id, **kwargs)
    
    def record_metrics(self, metrics: LogMetrics):
        """성능 메트릭 기록"""
        with self.metrics_lock:
            self.metrics_store.append(metrics)
            
            # 최대 1000개 메트릭만 메모리에 유지
            if len(self.metrics_store) > 1000:
                self.metrics_store = self.metrics_store[-500:]
        
        # 메트릭 로그 기록
        self.info(
            f"Performance metrics recorded",
            category=LogCategory.PERFORMANCE,
            request_id=metrics.request_id,
            **asdict(metrics)
        )
    
    def get_metrics_summary(self, category: Optional[str] = None, 
                           minutes: int = 60) -> Dict[str, Any]:
        """메트릭 요약 통계"""
        with self.metrics_lock:
            now = datetime.now(timezone.utc)
            cutoff = now.timestamp() - (minutes * 60)
            
            # 시간 범위 내 메트릭 필터링
            recent_metrics = [
                m for m in self.metrics_store
                if datetime.fromisoformat(m.timestamp.replace('Z', '+00:00')).timestamp() > cutoff
            ]
            
            # 카테고리 필터링
            if category:
                recent_metrics = [m for m in recent_metrics if m.category == category]
            
            if not recent_metrics:
                return {
                    "total_requests": 0,
                    "success_rate": 0.0,
                    "avg_duration_ms": 0.0,
                    "avg_memory_mb": 0.0,
                    "categories": {}
                }
            
            # 통계 계산
            total_requests = len(recent_metrics)
            successful_requests = sum(1 for m in recent_metrics if m.success)
            success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
            
            avg_duration = sum(m.duration_ms for m in recent_metrics) / total_requests
            avg_memory = sum(m.memory_mb for m in recent_metrics) / total_requests
            
            # 카테고리별 통계
            categories = {}
            for metric in recent_metrics:
                cat = metric.category
                if cat not in categories:
                    categories[cat] = {
                        "count": 0,
                        "success_count": 0,
                        "avg_duration_ms": 0.0,
                        "errors": []
                    }
                
                categories[cat]["count"] += 1
                if metric.success:
                    categories[cat]["success_count"] += 1
                else:
                    categories[cat]["errors"].append(metric.error_message)
            
            # 카테고리별 평균 계산
            for cat_data in categories.values():
                cat_metrics = [m for m in recent_metrics if m.category == cat]
                cat_data["avg_duration_ms"] = sum(m.duration_ms for m in cat_metrics) / len(cat_metrics)
                cat_data["success_rate"] = cat_data["success_count"] / cat_data["count"]
            
            return {
                "total_requests": total_requests,
                "success_rate": success_rate,
                "avg_duration_ms": round(avg_duration, 2),
                "avg_memory_mb": round(avg_memory, 2),
                "time_window_minutes": minutes,
                "categories": categories
            }


def performance_monitor(category: LogCategory = LogCategory.PERFORMANCE, 
                       operation: str = "unknown"):
    """성능 모니터링 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import psutil
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            request_id = getattr(kwargs.get('request'), 'state', {}).get('request_id', 'unknown')
            
            success = True
            error_message = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                duration_ms = (end_time - start_time) * 1000
                
                metrics = LogMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    request_id=request_id,
                    category=category.value,
                    operation=operation or func.__name__,
                    duration_ms=round(duration_ms, 2),
                    memory_mb=round(end_memory, 2),
                    cpu_percent=psutil.cpu_percent(),
                    success=success,
                    error_message=error_message
                )
                
                logger.record_metrics(metrics)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import psutil
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            success = True
            error_message = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                duration_ms = (end_time - start_time) * 1000
                
                metrics = LogMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    request_id="sync_operation",
                    category=category.value,
                    operation=operation or func.__name__,
                    duration_ms=round(duration_ms, 2),
                    memory_mb=round(end_memory, 2),
                    cpu_percent=psutil.cpu_percent(),
                    success=success,
                    error_message=error_message
                )
                
                logger.record_metrics(metrics)
        
        # async 함수인지 확인
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_api_call(category: LogCategory = LogCategory.API):
    """API 호출 로깅 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = None
            for arg in args:
                if hasattr(arg, 'method') and hasattr(arg, 'url'):
                    request = arg
                    break
            
            request_id = getattr(request.state, 'request_id', 'unknown') if request else 'unknown'
            
            logger.info(
                f"API call started: {func.__name__}",
                category=category,
                request_id=request_id,
                method=request.method if request else "unknown",
                path=request.url.path if request else "unknown"
            )
            
            try:
                result = await func(*args, **kwargs)
                logger.info(
                    f"API call completed: {func.__name__}",
                    category=category,
                    request_id=request_id
                )
                return result
            except Exception as e:
                logger.error(
                    f"API call failed: {func.__name__} - {str(e)}",
                    category=category,
                    request_id=request_id,
                    error_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator


# 전역 로거 인스턴스
logger = TrendyLyricsLogger()

# 표준 로깅을 위한 컨텍스트 매니저
class LogContext:
    """로그 컨텍스트 관리"""
    
    def __init__(self, request_id: str, category: LogCategory = LogCategory.SYSTEM):
        self.request_id = request_id
        self.category = category
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(
                f"Context ended with exception: {exc_type.__name__}",
                category=self.category,
                request_id=self.request_id,
                exc_info=(exc_type, exc_val, exc_tb)
            )
    
    def info(self, message: str, **kwargs):
        logger.info(message, self.category, self.request_id, **kwargs)
    
    def warning(self, message: str, **kwargs):
        logger.warning(message, self.category, self.request_id, **kwargs)
    
    def error(self, message: str, **kwargs):
        logger.error(message, self.category, self.request_id, **kwargs)
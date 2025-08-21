"""
Hugging Face 모델 캐싱 및 최적화 관리
메모리 효율적인 모델 로딩 및 캐시 전략
"""

import logging
import os
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import json
import threading
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# 안전한 import
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    )
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch/Transformers를 사용할 수 없습니다")
    TORCH_AVAILABLE = False
    torch = None


class ModelSize(Enum):
    """모델 크기 분류"""
    SMALL = "small"
    MEDIUM = "medium" 
    LARGE = "large"


@dataclass
class ModelMetadata:
    """모델 메타데이터"""
    name: str
    size: ModelSize
    last_used: float
    load_time: float
    memory_usage: int  # bytes
    success_count: int
    error_count: int


class HuggingFaceModelCache:
    """Hugging Face 모델 캐싱 시스템"""
    
    def __init__(self, cache_dir: str = "data/model_cache", max_memory_mb: int = 2048):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_usage = 0
        
        # 메모리 내 모델 캐시
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        self.loaded_extractors: Dict[str, Any] = {}
        
        # 메타데이터 관리
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.metadata_file = self.cache_dir / "metadata.json"
        
        # 스레드 안전성
        self.lock = threading.RLock()
        
        # 설정
        self.auto_cleanup = True
        self.cleanup_threshold = 0.8  # 80% 메모리 사용시 정리
        
        # 모델 우선순위 (작은 모델 우선)
        self.model_priority = {
            "cardiffnlp/twitter-roberta-base-emotion-multilingual-latest": ModelSize.SMALL,
            "j-hartmann/emotion-english-distilroberta-base": ModelSize.SMALL,
            "superb/hubert-large-superb-er": ModelSize.LARGE,
            "beomi/KcELECTRA-base-v2022": ModelSize.MEDIUM
        }
        
        self._load_metadata()
        logger.info(f"모델 캐시 초기화: {self.cache_dir}, 최대 메모리: {max_memory_mb}MB")
    
    def _load_metadata(self):
        """메타데이터 로드"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    
                for name, meta_dict in data.items():
                    self.model_metadata[name] = ModelMetadata(
                        name=meta_dict['name'],
                        size=ModelSize(meta_dict['size']),
                        last_used=meta_dict['last_used'],
                        load_time=meta_dict['load_time'],
                        memory_usage=meta_dict['memory_usage'],
                        success_count=meta_dict['success_count'],
                        error_count=meta_dict['error_count']
                    )
                    
                logger.info(f"메타데이터 로드 완료: {len(self.model_metadata)}개 모델")
        except Exception as e:
            logger.warning(f"메타데이터 로드 실패: {e}")
    
    def _save_metadata(self):
        """메타데이터 저장"""
        try:
            data = {}
            for name, metadata in self.model_metadata.items():
                data[name] = {
                    'name': metadata.name,
                    'size': metadata.size.value,
                    'last_used': metadata.last_used,
                    'load_time': metadata.load_time,
                    'memory_usage': metadata.memory_usage,
                    'success_count': metadata.success_count,
                    'error_count': metadata.error_count
                }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"메타데이터 저장 실패: {e}")
    
    def _estimate_model_memory(self, model_name: str) -> int:
        """모델 메모리 사용량 추정"""
        size = self.model_priority.get(model_name, ModelSize.MEDIUM)
        
        size_estimates = {
            ModelSize.SMALL: 300 * 1024 * 1024,   # 300MB
            ModelSize.MEDIUM: 500 * 1024 * 1024,  # 500MB
            ModelSize.LARGE: 1200 * 1024 * 1024   # 1.2GB
        }
        
        return size_estimates.get(size, 500 * 1024 * 1024)
    
    def _can_load_model(self, model_name: str) -> bool:
        """모델 로드 가능 여부 확인"""
        estimated_memory = self._estimate_model_memory(model_name)
        
        if self.current_memory_usage + estimated_memory > self.max_memory_bytes:
            if self.auto_cleanup:
                self._cleanup_old_models(estimated_memory)
                return self.current_memory_usage + estimated_memory <= self.max_memory_bytes
            return False
        
        return True
    
    def _cleanup_old_models(self, required_memory: int):
        """오래된 모델 정리"""
        logger.info("메모리 부족으로 모델 정리 시작...")
        
        # 사용 시간순으로 정렬 (오래된 것부터)
        sorted_models = sorted(
            self.model_metadata.items(),
            key=lambda x: x[1].last_used
        )
        
        freed_memory = 0
        models_to_remove = []
        
        for model_name, metadata in sorted_models:
            if model_name in self.loaded_models:
                models_to_remove.append(model_name)
                freed_memory += metadata.memory_usage
                
                if freed_memory >= required_memory:
                    break
        
        # 모델 언로드
        for model_name in models_to_remove:
            self._unload_model(model_name)
            logger.info(f"모델 언로드: {model_name}")
    
    def _unload_model(self, model_name: str):
        """모델 언로드"""
        with self.lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
            
            if model_name in self.loaded_tokenizers:
                del self.loaded_tokenizers[model_name]
            
            if model_name in self.loaded_extractors:
                del self.loaded_extractors[model_name]
            
            # 메모리 사용량 업데이트
            if model_name in self.model_metadata:
                self.current_memory_usage -= self.model_metadata[model_name].memory_usage
            
            # GPU 메모리 정리 (PyTorch 사용 시)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def load_text_model(self, model_name: str, force_reload: bool = False) -> Tuple[Optional[Any], Optional[Any]]:
        """텍스트 모델 로드 (토크나이저 포함)"""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch를 사용할 수 없어 모델을 로드할 수 없습니다")
            return None, None
        
        with self.lock:
            # 이미 로드된 경우
            if not force_reload and model_name in self.loaded_models:
                self._update_usage(model_name)
                return self.loaded_models[model_name], self.loaded_tokenizers.get(model_name)
            
            # 메모리 확인
            if not self._can_load_model(model_name):
                logger.warning(f"메모리 부족으로 모델 로드 불가: {model_name}")
                return None, None
            
            try:
                start_time = time.time()
                logger.info(f"텍스트 모델 로딩 중: {model_name}")
                
                # 토크나이저 로드
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir / "tokenizers")
                )
                
                # 모델 로드
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir / "models")
                )
                
                # GPU로 이동 (가능한 경우)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                model.eval()  # 평가 모드
                
                load_time = time.time() - start_time
                estimated_memory = self._estimate_model_memory(model_name)
                
                # 캐시에 저장
                self.loaded_models[model_name] = model
                self.loaded_tokenizers[model_name] = tokenizer
                self.current_memory_usage += estimated_memory
                
                # 메타데이터 업데이트
                self._update_metadata(model_name, load_time, estimated_memory, success=True)
                
                logger.info(f"모델 로딩 완료: {model_name} ({load_time:.2f}초)")
                return model, tokenizer
                
            except Exception as e:
                logger.error(f"모델 로딩 실패: {model_name} - {e}")
                self._update_metadata(model_name, 0, 0, success=False)
                return None, None
    
    async def load_audio_model(self, model_name: str, force_reload: bool = False) -> Tuple[Optional[Any], Optional[Any]]:
        """오디오 모델 로드 (특성 추출기 포함)"""
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch를 사용할 수 없어 모델을 로드할 수 없습니다")
            return None, None
        
        with self.lock:
            # 이미 로드된 경우
            if not force_reload and model_name in self.loaded_models:
                self._update_usage(model_name)
                return self.loaded_models[model_name], self.loaded_extractors.get(model_name)
            
            # 메모리 확인
            if not self._can_load_model(model_name):
                logger.warning(f"메모리 부족으로 모델 로드 불가: {model_name}")
                return None, None
            
            try:
                start_time = time.time()
                logger.info(f"오디오 모델 로딩 중: {model_name}")
                
                # 특성 추출기 로드
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir / "extractors")
                )
                
                # 모델 로드
                model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir / "models")
                )
                
                # GPU로 이동 (가능한 경우)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                model.eval()
                
                load_time = time.time() - start_time
                estimated_memory = self._estimate_model_memory(model_name)
                
                # 캐시에 저장
                self.loaded_models[model_name] = model
                self.loaded_extractors[model_name] = feature_extractor
                self.current_memory_usage += estimated_memory
                
                # 메타데이터 업데이트
                self._update_metadata(model_name, load_time, estimated_memory, success=True)
                
                logger.info(f"모델 로딩 완료: {model_name} ({load_time:.2f}초)")
                return model, feature_extractor
                
            except Exception as e:
                logger.error(f"모델 로딩 실패: {model_name} - {e}")
                self._update_metadata(model_name, 0, 0, success=False)
                return None, None
    
    def _update_usage(self, model_name: str):
        """모델 사용 시간 업데이트"""
        if model_name in self.model_metadata:
            self.model_metadata[model_name].last_used = time.time()
            self.model_metadata[model_name].success_count += 1
    
    def _update_metadata(self, model_name: str, load_time: float, memory_usage: int, success: bool):
        """메타데이터 업데이트"""
        if model_name not in self.model_metadata:
            size = self.model_priority.get(model_name, ModelSize.MEDIUM)
            self.model_metadata[model_name] = ModelMetadata(
                name=model_name,
                size=size,
                last_used=time.time(),
                load_time=load_time,
                memory_usage=memory_usage,
                success_count=1 if success else 0,
                error_count=0 if success else 1
            )
        else:
            metadata = self.model_metadata[model_name]
            metadata.last_used = time.time()
            if success:
                metadata.success_count += 1
                if load_time > 0:
                    metadata.load_time = load_time
                if memory_usage > 0:
                    metadata.memory_usage = memory_usage
            else:
                metadata.error_count += 1
        
        # 주기적으로 메타데이터 저장
        if time.time() % 60 < 1:  # 대략 1분마다
            self._save_metadata()
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계 반환"""
        with self.lock:
            return {
                'loaded_models': list(self.loaded_models.keys()),
                'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                'memory_limit_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_usage_percent': (self.current_memory_usage / self.max_memory_bytes) * 100,
                'total_models_seen': len(self.model_metadata),
                'cache_hit_rate': self._calculate_hit_rate()
            }
    
    def _calculate_hit_rate(self) -> float:
        """캐시 히트율 계산"""
        total_requests = sum(meta.success_count + meta.error_count for meta in self.model_metadata.values())
        cache_hits = len([name for name in self.loaded_models.keys() if name in self.model_metadata])
        
        return (cache_hits / max(total_requests, 1)) * 100
    
    def cleanup_all(self):
        """모든 모델 정리"""
        with self.lock:
            logger.info("모든 모델 정리 중...")
            
            for model_name in list(self.loaded_models.keys()):
                self._unload_model(model_name)
            
            self._save_metadata()
            logger.info("모델 정리 완료")
    
    def get_recommended_models(self) -> List[str]:
        """추천 모델 목록 (성능 기반)"""
        # 성공률과 로딩 시간을 고려한 점수 계산
        scored_models = []
        
        for name, metadata in self.model_metadata.items():
            if metadata.success_count > 0:
                success_rate = metadata.success_count / (metadata.success_count + metadata.error_count)
                load_time_score = max(0, 1 - (metadata.load_time / 60))  # 1분 이상이면 0점
                
                # 작은 모델에 보너스
                size_bonus = {
                    ModelSize.SMALL: 0.3,
                    ModelSize.MEDIUM: 0.1,
                    ModelSize.LARGE: 0.0
                }.get(metadata.size, 0.0)
                
                score = success_rate * 0.6 + load_time_score * 0.3 + size_bonus
                scored_models.append((name, score))
        
        # 점수순으로 정렬
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, score in scored_models[:5]]


# 전역 모델 캐시 인스턴스
model_cache = HuggingFaceModelCache()
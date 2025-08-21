"""
의존성 및 모델 로딩 유틸리티
누락된 라이브러리 처리 및 우아한 대체 방안 제공
"""

import logging
import importlib
import sys
from typing import Optional, Dict, Any, Callable
from functools import wraps
import warnings

logger = logging.getLogger(__name__)


class DependencyManager:
    """의존성 관리 및 대체 방안 제공"""
    
    def __init__(self):
        self.available_packages = {}
        self.fallback_strategies = {}
        self._check_core_dependencies()
    
    def _check_core_dependencies(self):
        """핵심 의존성 확인"""
        core_deps = {
            'torch': 'PyTorch for Hugging Face models',
            'transformers': 'Hugging Face transformers',
            'librosa': 'Audio processing',
            'numpy': 'Numerical computations',
            'sklearn': 'Machine learning utilities'
        }
        
        for package, description in core_deps.items():
            self.available_packages[package] = self._check_package(package)
            if not self.available_packages[package]:
                logger.warning(f"선택적 의존성 누락: {package} ({description})")
    
    def _check_package(self, package_name: str) -> bool:
        """패키지 설치 여부 확인"""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    def require_package(self, package_name: str, fallback_func: Optional[Callable] = None):
        """패키지 필수 검증 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.available_packages.get(package_name, False):
                    logger.warning(f"패키지 {package_name}이(가) 없습니다. 대체 방안을 사용합니다.")
                    if fallback_func:
                        return fallback_func(*args, **kwargs)
                    else:
                        raise ImportError(f"필수 패키지 {package_name}이(가) 설치되지 않았습니다.")
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_safe_import(self, module_name: str, class_name: str = None):
        """안전한 import with 대체 방안"""
        try:
            module = importlib.import_module(module_name)
            if class_name:
                return getattr(module, class_name)
            return module
        except ImportError as e:
            logger.warning(f"모듈 import 실패: {module_name} - {e}")
            return None
    
    def install_missing_dependencies(self, packages: list):
        """누락된 의존성 자동 설치 시도"""
        import subprocess
        
        for package in packages:
            if not self.available_packages.get(package, False):
                try:
                    logger.info(f"패키지 {package} 설치 시도 중...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    self.available_packages[package] = True
                    logger.info(f"패키지 {package} 설치 완료")
                except subprocess.CalledProcessError as e:
                    logger.error(f"패키지 {package} 설치 실패: {e}")


# 전역 의존성 관리자
dependency_manager = DependencyManager()


def safe_import_with_fallback(primary_import: str, fallback_func: Callable = None):
    """안전한 import with 대체 함수"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ImportError as e:
                logger.warning(f"Import 오류: {e}")
                if fallback_func:
                    logger.info("대체 함수를 사용합니다.")
                    return await fallback_func(*args, **kwargs)
                else:
                    return {'error': f'모듈 로딩 실패: {primary_import}', 'fallback': 'none'}
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                logger.warning(f"Import 오류: {e}")
                if fallback_func:
                    logger.info("대체 함수를 사용합니다.")
                    return fallback_func(*args, **kwargs)
                else:
                    return {'error': f'모듈 로딩 실패: {primary_import}', 'fallback': 'none'}
        
        # async 함수인지 확인
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def check_gpu_availability() -> Dict[str, Any]:
    """GPU 사용 가능성 확인"""
    gpu_info = {
        'cuda_available': False,
        'mps_available': False,  # Apple Silicon
        'device_count': 0,
        'recommended_device': 'cpu'
    }
    
    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        gpu_info['device_count'] = torch.cuda.device_count() if gpu_info['cuda_available'] else 0
        
        # Apple Silicon MPS 지원 확인
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info['mps_available'] = True
            gpu_info['recommended_device'] = 'mps'
        elif gpu_info['cuda_available']:
            gpu_info['recommended_device'] = 'cuda'
            
    except ImportError:
        logger.warning("PyTorch가 설치되지 않아 GPU 정보를 확인할 수 없습니다.")
    
    return gpu_info


def optimize_model_loading():
    """모델 로딩 최적화 설정"""
    try:
        import torch
        
        # 메모리 최적화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Warning 억제 (선택적)
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        
        return True
    except ImportError:
        return False


def get_fallback_emotion_analysis():
    """Hugging Face 모델 실패시 대체 감정 분석"""
    
    # 간단한 키워드 기반 감정 분석
    positive_keywords = ['사랑', '행복', '기쁨', '좋아', '웃음', '희망', '꿈', '함께', '아름다운', '따뜻한']
    negative_keywords = ['슬픔', '아픔', '눈물', '이별', '외로운', '그리움', '힘들어', '아파', '슬픈', '어둠']
    
    async def fallback_text_emotion(text: str) -> Dict:
        """키워드 기반 텍스트 감정 분석"""
        if not text:
            return {
                'emotions': {'neutral': 1.0},
                'dominant_emotion': 'neutral',
                'confidence': 0.3,
                'korean_emotion': '중립적',
                'method': 'keyword_fallback'
            }
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text)
        
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return {
                'emotions': {'neutral': 1.0},
                'dominant_emotion': 'neutral',
                'confidence': 0.3,
                'korean_emotion': '중립적',
                'method': 'keyword_fallback'
            }
        
        positive_ratio = positive_count / total_keywords
        negative_ratio = negative_count / total_keywords
        
        if positive_ratio > negative_ratio:
            dominant = 'positive'
            korean_emotion = '긍정적'
            confidence = min(positive_ratio + 0.3, 0.8)
        elif negative_ratio > positive_ratio:
            dominant = 'negative'
            korean_emotion = '부정적'
            confidence = min(negative_ratio + 0.3, 0.8)
        else:
            dominant = 'neutral'
            korean_emotion = '중립적'
            confidence = 0.5
        
        emotions = {
            'positive': positive_ratio,
            'negative': negative_ratio,
            'neutral': 1 - (positive_ratio + negative_ratio)
        }
        
        return {
            'emotions': emotions,
            'dominant_emotion': dominant,
            'confidence': confidence,
            'korean_emotion': korean_emotion,
            'method': 'keyword_fallback',
            'keyword_counts': {
                'positive': positive_count,
                'negative': negative_count
            }
        }
    
    return fallback_text_emotion


def install_requirements_on_demand():
    """필요에 따라 requirements 설치"""
    
    missing_packages = []
    
    # 필수 패키지 확인
    required_packages = [
        ('transformers', '4.35.0'),
        ('torch', '2.0.0'),
        ('librosa', '0.10.2'),
        ('scikit-learn', '1.3.0')
    ]
    
    for package, min_version in required_packages:
        if not dependency_manager._check_package(package):
            missing_packages.append(f"{package}>={min_version}")
    
    if missing_packages:
        logger.info(f"누락된 패키지 감지: {missing_packages}")
        
        install_command = f"pip install {' '.join(missing_packages)}"
        print(f"📦 다음 명령어로 누락된 패키지를 설치하세요:")
        print(f"   {install_command}")
        
        # 자동 설치 시도 (선택적)
        response = input("지금 자동으로 설치하시겠습니까? (y/n): ").lower()
        if response == 'y':
            dependency_manager.install_missing_dependencies([p.split('>=')[0] for p in missing_packages])
    
    return len(missing_packages) == 0
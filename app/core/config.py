from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    This uses pydantic-settings (pydantic v2) and reads variables from a local
    .env file if present.
    """

    # pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "Trendy Lyrics API"
    VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    ENV: str = "development"
    DEBUG: bool = False

    # 한국어 기본 설정
    DEFAULT_LANGUAGE: str = "ko"
    DEFAULT_REGION: str = "KR"
    
    # Optional API keys / feature flags
    YOUTUBE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"
    ENABLE_OPENAI_EMBEDDINGS: bool = False
    ENABLE_TRANSCRIBE: bool = False
    ENABLE_AUDIO_EMOTION: bool = False
    ENABLE_PITCH_ANALYSIS: bool = False

    # Processing
    TEXT_EMBED_DIM: int = 256

    # External integration
    SPRING_CALLBACK_URL: str = ""
    # Qdrant (optional)
    ENABLE_QDRANT: bool = False
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    # Celery/Redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_ALWAYS_EAGER: bool = False

    # Transcribe defaults (한국어 최적화)
    TRANSCRIBE_MODEL_SIZE: str = "small"  # 한국어 인식률 향상을 위해 small 이상 권장
    TRANSCRIBE_COMPUTE_TYPE: str = "default"
    TRANSCRIBE_LANGUAGE: str = "ko"  # 기본 한국어

    # Audio analysis (한국어 음성 특성에 맞춤)
    PITCH_MIN_NOTE: str = "C2"  # 한국어 남성 음역대
    PITCH_MAX_NOTE: str = "C7"  # 한국어 여성 음역대
    EMOTION_MODEL_ID: str = "superb/hubert-large-superb-er"  # 한국어 감정 인식 모델
    
    # 한국어 감정 카테고리
    KOREAN_EMOTIONS: list = ["신뢰감", "친근함", "활기참", "차분함", "중립", "열정", "우아함"]

    # Suno defaults (콜백용 페이로드 빌드에 사용)
    SUNO_MODEL: str = "V3_5"  # V3_5 | V4 | V4_5
    SUNO_DEFAULT_CUSTOM_MODE: bool = True
    SUNO_DEFAULT_INSTRUMENTAL: bool = False
    SUNO_DEFAULT_STYLE_WEIGHT: float = 0.65
    SUNO_DEFAULT_WEIRDNESS: float = 0.65
    SUNO_DEFAULT_AUDIO_WEIGHT: float = 0.65
    SUNO_DEFAULT_VOCAL_GENDER: str = ""  # "m" | "f" | ""
    SUNO_NEGATIVE_TAGS: str = ""  # 쉼표 구분 문자열
    SPRING_SUNO_CALLBACK_URL: str = ""  # 스프링 서버가 Suno에 등록할 콜백 URL


settings = Settings()



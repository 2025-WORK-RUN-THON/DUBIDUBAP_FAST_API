from fastapi import APIRouter
from app.api.v1.videos import router as videos_router
from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.process import router as process_router
from app.api.v1.analyze import router as analyze_router
from app.api.v1.recommend import router as recommend_router
from app.api.v1.trends import router as trends_router
from app.api.v1.generate import router as generate_router
from app.api.v1.admin import router as admin_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.music import router as music_router
from app.api.v1.monitoring import router as monitoring_router


api_router = APIRouter()


@api_router.get("/ping", tags=["system"])
def ping():
    return {"message": "pong"}

api_router.include_router(monitoring_router)  # 모니터링 엔드포인트
api_router.include_router(music_router)  # 새로운 음악 분석 엔드포인트
api_router.include_router(videos_router)
api_router.include_router(embeddings_router)
api_router.include_router(process_router)
api_router.include_router(analyze_router)
api_router.include_router(recommend_router)
api_router.include_router(trends_router)
api_router.include_router(generate_router)
api_router.include_router(admin_router)
api_router.include_router(jobs_router)



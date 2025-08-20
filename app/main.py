from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
import time
import logging
import uuid
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.api.v1.router import api_router
from app.db import init_db


def create_app() -> FastAPI:
    app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @app.get("/health", tags=["system"])  # simple health check
    def health():
        return {"status": "ok", "env": settings.ENV}

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = rid
        start = time.time()
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        logging.info("%s %s -> %s (%d ms) rid=%s", request.method, request.url.path, response.status_code, duration_ms, rid)
        response.headers["X-Request-Id"] = rid
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        # Standardize error body
        body = {
            "status": exc.status_code,
            "code": "HTTP_ERROR",
            "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            "path": request.url.path,
        }
        rid = getattr(request.state, "request_id", None)
        if rid:
            body["requestId"] = rid
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        body = {
            "status": 500,
            "code": "INTERNAL_SERVER_ERROR",
            "message": str(exc),
            "path": request.url.path,
        }
        rid = getattr(request.state, "request_id", None)
        if rid:
            body["requestId"] = rid
        return JSONResponse(status_code=500, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        body = {
            "status": 400,
            "code": "VALIDATION_ERROR",
            "message": "입력값 검증에 실패했습니다.",
            "path": request.url.path,
            "errors": exc.errors(),
        }
        rid = getattr(request.state, "request_id", None)
        if rid:
            body["requestId"] = rid
        return JSONResponse(status_code=400, content=body)

    @app.on_event("startup")
    def on_startup() -> None:
        # Ensure tables exist on startup
        init_db()

    return app


app = create_app()



"""FastAPI application factory for Aegis."""

from __future__ import annotations

import logging

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aegis.api.middleware import CorrelationIdMiddleware, RequestLoggingMiddleware
from aegis.api.routes import admin, feedback, scan, session
from aegis.core.config import get_config
from aegis.core.models import ErrorResponse, HealthResponse
from aegis.version import __version__

logger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title="Aegis Content Security API",
        description="Content security microservice with CrewAI agents detecting prompt injection attacks",
        version=__version__,
        docs_url="/docs" if config.app.debug else None,
        redoc_url="/redoc" if config.app.debug else None,
        openapi_url="/openapi.json" if config.app.debug else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if config.app.debug else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    app.include_router(scan.router)
    app.include_router(session.router)
    app.include_router(feedback.router)
    app.include_router(admin.router)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("aegis_starting", version=__version__, env=config.app.env)
        from aegis.core.cache import get_cache
        cache = get_cache()
        try:
            await cache.connect()
            logger.info("redis_connected")
        except Exception as exc:
            logger.warning("redis_connection_failed", error=str(exc))

        from aegis.storage.supabase import get_store
        store = get_store()
        logger.info("supabase_connected", url=config.supabase.url[:30] + "..." if config.supabase.url else "not configured")

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("aegis_shutting_down")
        from aegis.core.cache import get_cache
        cache = get_cache()
        await cache.disconnect()

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        from aegis.core.cache import get_cache
        cache = get_cache()

        services: dict[str, str] = {}

        try:
            redis_ok = await cache.ping()
            services["redis"] = "ok" if redis_ok else "degraded"
        except Exception:
            services["redis"] = "unavailable"

        try:
            from aegis.storage.supabase import get_store
            store = get_store()
            store.client.table("jobs").select("id").limit(1).execute()
            services["supabase"] = "ok"
        except Exception:
            services["supabase"] = "unavailable"

        return HealthResponse(
            status="ok",
            version=__version__,
            services=services,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.error("unhandled_exception", error=str(exc), request_id=request_id, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                request_id=request_id,
            ).model_dump(),
        )

    return app


app = create_app()

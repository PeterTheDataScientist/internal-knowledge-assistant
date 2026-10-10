"""Application entry point."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.errors import register_exception_handlers
from app.llm.factory import build_llm_provider
from app.observability.logging_config import configure_logging
from app.observability.middleware import RateLimitMiddleware, RequestContextMiddleware
from app.rag.knowledge_base import build_knowledge_base
from app.rag.memory import ConversationMemory
from app.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

DESCRIPTION = """
Retrieval augmented assistant over Contoso Financial internal documentation.

The service answers only from the indexed corpus, returns the source file for every
claim, and replies with `I don't have sufficient information in the provided documents.`
whenever the retrieved passages do not support an answer.

Guardrails run on the question, on the retrieved passages, and on the generated answer.
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    app.state.started_at = time.time()

    logger.info(
        "starting service",
        extra={"version": settings.app_version, "environment": settings.environment},
    )

    try:
        knowledge_base = build_knowledge_base(settings)
        llm = build_llm_provider(settings)
        app.state.pipeline = RAGPipeline(
            settings=settings,
            knowledge_base=knowledge_base,
            llm=llm,
            memory=ConversationMemory(
                settings.memory_max_turns,
                settings.memory_ttl_seconds,
                settings.memory_max_sessions,
            ),
        )
        logger.info("service ready")
    except Exception:
        # Startup failures leave the app running so that /health reports the fault to the
        # orchestrator instead of the container crash looping with no diagnostics.
        logger.exception("startup failed, service will report degraded health")
        app.state.pipeline = None

    yield

    logger.info("shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "Peter Tinashe Mundowa"},
        license_info={"name": "Proprietary"},
    )

    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index() -> dict:
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "endpoints": ["POST /ask", "POST /ask/stream", "GET /health", "GET /metrics", "GET /sources"],
        }

    return app


app = create_app()

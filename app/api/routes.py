"""HTTP routes."""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List

from fastapi import APIRouter, Depends, Header, Request, Response, status
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.errors import IndexNotReadyError, UnauthorisedError
from app.observability.metrics import metrics
from app.rag.pipeline import RAGPipeline
from app.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthComponent,
    HealthResponse,
    MetricsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Feedback is held in memory in this prototype. In production this is an append-only
# table, because thumbs-down events are the cheapest source of new evaluation cases.
_feedback_log: List[Dict[str, object]] = []


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise IndexNotReadyError()
    return pipeline


def require_api_key(
    settings: Settings = Depends(get_settings),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if not settings.require_api_key:
        return
    if not x_api_key or x_api_key not in settings.api_keys:
        raise UnauthorisedError()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["operations"],
    summary="Liveness and readiness",
)
def health(request: Request, settings: Settings = Depends(get_settings)) -> HealthResponse:
    pipeline: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
    started_at = getattr(request.app.state, "started_at", time.time())

    components: Dict[str, HealthComponent] = {}
    if pipeline is None:
        components["knowledge_base"] = HealthComponent(status="down", detail="index not built")
        return HealthResponse(
            status="degraded",
            version=settings.app_version,
            environment=settings.environment,
            uptime_seconds=round(time.time() - started_at, 2),
            documents=0,
            chunks=0,
            embedding_backend="none",
            llm_provider="none",
            llm_model="none",
            components=components,
        )

    stats = pipeline.stats()
    components["knowledge_base"] = HealthComponent(
        status="ok", detail=f"{stats['documents']} documents, {stats['chunks']} chunks"
    )
    components["vector_index"] = HealthComponent(status="ok", detail=str(stats["vector_backend"]))
    components["embeddings"] = HealthComponent(status="ok", detail=str(stats["embedder"]))
    components["llm"] = HealthComponent(
        status="ok" if pipeline.llm.name != "mock" else "degraded",
        detail=f"provider={pipeline.llm.name} model={pipeline.llm.model}",
    )

    overall = "ok" if all(c.status == "ok" for c in components.values()) else "degraded"
    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - started_at, 2),
        documents=int(stats["documents"]),
        chunks=int(stats["chunks"]),
        embedding_backend=str(stats["embedder"]),
        llm_provider=pipeline.llm.name,
        llm_model=pipeline.llm.model,
        components=components,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["assistant"],
    summary="Ask a question about Contoso Financial internal documentation",
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        422: {"model": ErrorResponse, "description": "Request body failed validation"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Index or model backend unavailable"},
        504: {"model": ErrorResponse, "description": "Model backend timed out"},
    },
)
def ask(
    payload: AskRequest,
    response: Response,
    pipeline: RAGPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
) -> AskResponse:
    result = pipeline.answer(payload.question, session_id=payload.session_id, top_k=payload.top_k)

    metrics.record_request(
        result.latency_ms,
        confidence=result.confidence.value,
        refused=not result.answered_from_context,
        refusal_reason=result.refusal_reason,
        injection_blocked=result.refusal_reason == "prompt_injection",
        confidential_blocked=result.refusal_reason == "confidential_request",
    )

    # A refusal is a correct outcome, not a client error, so the status stays 200 and
    # the caller reads answered_from_context. A header is added so gateways and log
    # pipelines can route on it without parsing the body.
    response.headers["x-answered-from-context"] = str(result.answered_from_context).lower()
    return result


@router.post(
    "/ask/stream",
    tags=["assistant"],
    summary="Ask a question and receive the verified answer as a server-sent event stream",
)
def ask_stream(
    payload: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
) -> StreamingResponse:
    # The answer is produced before the response starts, so a model outage on this route
    # still returns 503 rather than a truncated 200 with a traceback in the log.
    result = pipeline.answer(payload.question, session_id=payload.session_id, top_k=payload.top_k)

    def event_source():
        final = None
        for piece in pipeline.stream_prepared(result):
            if isinstance(piece, tuple):
                final = piece[1]
                break
            yield f"event: token\ndata: {json.dumps({'text': piece})}\n\n"

        if final is not None:
            metrics.record_request(
                final.latency_ms,
                confidence=final.confidence.value,
                refused=not final.answered_from_context,
                refusal_reason=final.refusal_reason,
                injection_blocked=final.refusal_reason == "prompt_injection",
                confidential_blocked=final.refusal_reason == "confidential_request",
            )
            yield f"event: done\ndata: {final.model_dump_json()}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["assistant"],
    summary="Record whether an answer was useful",
)
def feedback(payload: FeedbackRequest, _: None = Depends(require_api_key)) -> FeedbackResponse:
    _feedback_log.append(
        {
            "request_id": payload.request_id,
            "helpful": payload.helpful,
            "comment": payload.comment,
            "recorded_at": time.time(),
        }
    )
    metrics.record_feedback(payload.helpful)
    logger.info(
        "feedback recorded",
        extra={"request_id": payload.request_id, "helpful": payload.helpful},
    )
    return FeedbackResponse(recorded=True, request_id=payload.request_id)


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["operations"],
    summary="Service metrics snapshot",
)
def metrics_snapshot() -> MetricsResponse:
    return MetricsResponse(**{
        key: value
        for key, value in metrics.snapshot().items()
        if key in MetricsResponse.model_fields
    })


@router.get("/sources", tags=["operations"], summary="List indexed documents")
def sources(pipeline: RAGPipeline = Depends(get_pipeline)) -> Dict[str, object]:
    return {
        "documents": [
            {
                "source": document.source,
                "title": document.title,
                "domain": document.domain,
                "trust": document.trust.value,
                "checksum": document.checksum[:12],
            }
            for document in pipeline.kb.documents
        ],
        "chunks": pipeline.kb.chunk_count,
    }

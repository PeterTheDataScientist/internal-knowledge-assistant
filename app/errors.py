"""Application exceptions and the handlers that turn them into JSON responses."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.observability.metrics import metrics

logger = logging.getLogger(__name__)


class AssistantError(Exception):
    """Base class for errors the API knows how to report."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "internal_error"
    message = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        if message:
            self.message = message


class IndexNotReadyError(AssistantError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "index_not_ready"
    message = "The knowledge index is not available yet."


class LLMUnavailableError(AssistantError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "llm_unavailable"
    message = "The language model backend is unavailable."


class LLMTimeoutError(AssistantError):
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    code = "llm_timeout"
    message = "The language model did not respond in time."


class RateLimitError(AssistantError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    message = "Too many requests. Try again shortly."


class UnauthorisedError(AssistantError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorised"
    message = "A valid API key is required."


def _payload(code: str, message: str, request_id: str | None, fields=None) -> dict:
    body = {"error": {"code": code, "message": message, "request_id": request_id}}
    if fields:
        body["error"]["fields"] = fields
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AssistantError)
    async def _assistant_error(request: Request, exc: AssistantError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "handled application error",
            extra={"error_code": exc.code, "request_id": request_id},
        )
        metrics.record_error()
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.code, exc.message, request_id),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        fields = [
            {
                "field": ".".join(str(part) for part in err.get("loc", []) if part != "body"),
                "reason": err.get("msg", "invalid value"),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_payload(
                "validation_error", "The request body failed validation.", request_id, fields
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        code_map = {
            400: "bad_request",
            401: "unauthorised",
            404: "not_found",
            405: "method_not_allowed",
            413: "payload_too_large",
            429: "rate_limited",
        }
        code = code_map.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(code, str(exc.detail), request_id),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unhandled error", extra={"request_id": request_id})
        metrics.record_error()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_payload(
                "internal_error",
                "An unexpected error occurred. The incident has been logged.",
                request_id,
            ),
        )

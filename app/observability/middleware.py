"""Request scoped middleware: correlation id, access logging, and rate limiting."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("app.access")

EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json", "/"}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attaches a correlation id and emits one structured access log per request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()

        response = await call_next(request)

        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = str(duration_ms)
        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed window per-client limiter.

    Suitable for a single node. Behind a load balancer this moves to the API gateway or
    to a shared Redis counter, because per-process limits do not compose.
    """

    def __init__(self, app, requests_per_minute: int = 30) -> None:
        super().__init__(app)
        self.limit = requests_per_minute
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        identity = request.headers.get("x-api-key") or (
            request.client.host if request.client else "anonymous"
        )
        now = time.time()

        with self._lock:
            # Drop clients that have been quiet for a full window, otherwise this map
            # grows for the lifetime of the process, one entry per address ever seen.
            if len(self._hits) > 10_000:
                for key in [k for k, v in self._hits.items() if not v or now - v[-1] > 120]:
                    del self._hits[key]

            window = self._hits[identity]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.limit:
                retry_after = max(1, int(60 - (now - window[0])))
                logger.warning("rate limit exceeded", extra={"identity_hash": hash(identity) % 10**8})
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "rate_limited",
                            "message": f"Rate limit of {self.limit} requests per minute exceeded.",
                            "request_id": getattr(request.state, "request_id", None),
                        }
                    },
                    headers={"retry-after": str(retry_after)},
                )
            window.append(now)

        return await call_next(request)

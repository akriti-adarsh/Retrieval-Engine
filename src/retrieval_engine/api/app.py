"""The FastAPI application.

One exception handler produces every error body, so a client can switch on ``error.code``
without discovering that some routes fail differently from others. That is only possible
because every deliberate error in this service subclasses ``RetrievalEngineError`` and
carries its own code and status.

The service graph is built in the lifespan and stored on ``app.state``, not in a module
global. Tests build several applications in one process, and a global would leak one test's
store into the next.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from retrieval_engine import __version__
from retrieval_engine.api.deps import Services, build_services
from retrieval_engine.api.routes_admin import (
    REQUEST_LATENCY,
    REQUESTS,
    documents_router,
)
from retrieval_engine.api.routes_admin import router as admin_router
from retrieval_engine.api.routes_ingest import router as ingest_router
from retrieval_engine.api.routes_query import router as query_router
from retrieval_engine.config import Settings, get_settings
from retrieval_engine.errors import (
    RateLimitExceededError,
    RetrievalEngineError,
    StoreUnavailableError,
)
from retrieval_engine.logging_config import (
    bind_request_id,
    clear_request_context,
    configure_logging,
    get_logger,
    new_request_id,
)
from retrieval_engine.models import ErrorDetail, ErrorResponse

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

DESCRIPTION = """\
Retrieval-augmented generation over an indexed corpus: hybrid dense and BM25 retrieval,
reciprocal rank fusion, cross-encoder reranking, and grounding verification.

A low-confidence question returns 200 with `answer_type: "refused"`. Refusing is a correct
outcome, not an error.
"""


#: Paths exempt from rate limiting. A limiter that can starve the liveness probe will take a
#: healthy service out of rotation under exactly the load it was meant to protect it from.
UNLIMITED_PATHS = frozenset({"/health", "/health/ready", "/metrics"})


@dataclass
class TokenBucket:
    """One caller's allowance. Refills continuously rather than resetting on a boundary.

    A fixed window lets a caller spend its whole allowance at 59.9 seconds and again at 60.1,
    which is twice the intended rate at the worst possible moment. Continuous refill has no
    such edge.
    """

    capacity: float
    refill_per_second: float
    tokens: float = field(default=0.0)
    updated_at: float = field(default=0.0)
    #: An explicit flag rather than treating updated_at == 0.0 as "never used". A monotonic
    #: clock legitimately reads 0.0, and using that as a sentinel refilled the bucket to full
    #: on every call, which silently disabled the limiter for the first caller.
    started: bool = field(default=False)

    def take(self, now: float) -> bool:
        """Spend one token if the bucket has it, returning whether the request may proceed."""
        if not self.started:
            self.started = True
            self.tokens = self.capacity
            self.updated_at = now
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.updated_at = now
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True


class RateLimiter:
    """Per-caller token buckets, with a clock injected so the tests are not timing-dependent."""

    def __init__(
        self,
        per_minute: int,
        *,
        clock: Callable[[], float] | None = None,
        max_buckets: int = 10_000,
    ) -> None:
        self._capacity = float(per_minute)
        self._refill = per_minute / 60.0
        self._clock = clock if clock is not None else time.monotonic
        self._max_buckets = max_buckets
        self._buckets: dict[str, TokenBucket] = {}

    def allow(self, key: str) -> bool:
        """Whether ``key`` may make a request now."""
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self._max_buckets:
                # Unbounded per-IP state is a memory leak an attacker controls. Dropping the
                # fullest buckets first keeps the callers who are actually being limited.
                self._evict(now)
            bucket = TokenBucket(capacity=self._capacity, refill_per_second=self._refill)
            self._buckets[key] = bucket
        return bucket.take(now)

    def _evict(self, now: float) -> None:
        """Drop buckets that have refilled to full, since they carry no useful state."""
        for key, bucket in list(self._buckets.items()):
            elapsed = max(0.0, now - bucket.updated_at)
            if bucket.tokens + elapsed * self._refill >= self._capacity:
                del self._buckets[key]
        if len(self._buckets) >= self._max_buckets:  # pragma: no cover - pathological load
            self._buckets.clear()


def client_key(request: Request) -> str:
    """Identify the caller for rate limiting.

    Deliberately the socket peer, not a forwarded header: X-Forwarded-For is caller-supplied
    and trusting it lets anyone bypass the limiter by varying one header. Behind a real proxy
    this needs the proxy's own trusted forwarding, which is deployment configuration rather
    than something this code can safely assume.
    """
    return request.client.host if request.client else "unknown"


def _error_response(request: Request, code: str, message: str, http_status: int) -> JSONResponse:
    """Build the one error shape this API returns."""
    request_id = getattr(request.state, "request_id", "")
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id if isinstance(request_id, str) else "",
        )
    )
    return JSONResponse(
        status_code=http_status,
        content=body.model_dump(),
        headers={REQUEST_ID_HEADER: body.error.request_id} if body.error.request_id else None,
    )


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
    *,
    clock: Callable[[], float] | None = None,
) -> FastAPI:
    """Build the application.

    ``services`` is the injection seam: tests pass a graph wired to an in-memory store and
    fakes, so the API can be exercised without a database, a model download, or a network.
    ``clock`` is the seam for the rate limiter, so its tests assert behaviour rather than
    sleeping and hoping.
    """
    active = (
        settings if settings is not None else (services.settings if services else get_settings())
    )
    configure_logging(active.log_level, json_logs=active.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        graph = services if services is not None else build_services(active)
        app.state.services = graph
        try:
            # Create the collection up front so the first query does not race ingestion.
            # A store that is down must not stop the app from starting: /health/ready is
            # what reports that, and it can only report it if the app is running.
            await graph.store.ensure_collection(graph.embedder.info)
        except StoreUnavailableError as exc:
            logger.warning("store_unavailable_at_startup", error=exc.message)
        logger.info("application_started", version=__version__, env=active.env)
        try:
            yield
        finally:
            await graph.aclose()
            logger.info("application_stopped")

    app = FastAPI(
        title=active.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    limiter = RateLimiter(active.rate_limit_per_minute, clock=clock)
    app.state.rate_limiter = limiter

    @app.middleware("http")
    async def observe_and_limit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Rate limit, then time the request and count it by route.

        The route template is used as the label, not the raw path, so a path parameter cannot
        create unbounded label cardinality and blow up the metrics store.
        """
        path = request.url.path
        if (
            active.rate_limit_enabled
            and path not in UNLIMITED_PATHS
            and not limiter.allow(client_key(request))
        ):
            error = RateLimitExceededError(
                f"rate limit of {active.rate_limit_per_minute} requests per minute exceeded"
            )
            REQUESTS.labels(route=path, method=request.method, status=error.http_status).inc()
            return _error_response(request, error.code, error.message, error.http_status)

        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        label = getattr(route, "path", path)
        REQUEST_LATENCY.labels(route=label).observe(time.perf_counter() - started)
        REQUESTS.labels(route=label, method=request.method, status=response.status_code).inc()
        return response

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Accept an inbound request id or mint one, and bind it for the whole request.

        The id goes into a contextvar, so a log line emitted deep inside the retrieval
        pipeline carries it without every function signature having to pass it along.
        """
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming if incoming else new_request_id()
        request.state.request_id = request_id
        bind_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            clear_request_context()
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    @app.exception_handler(RetrievalEngineError)
    async def handle_known(request: Request, exc: Exception) -> JSONResponse:
        """Every deliberate error, mapped by its own declared code and status."""
        error = exc if isinstance(exc, RetrievalEngineError) else RetrievalEngineError(str(exc))
        logger.warning("request_failed", code=error.code, message=error.message)
        return _error_response(request, error.code, error.message, error.http_status)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: Exception) -> JSONResponse:
        """Schema violations, in the same envelope as everything else."""
        detail = str(exc)
        return _error_response(request, "validation_error", detail, 422)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        """The last resort. The message is generic; the detail goes to the log.

        Returning the exception text would leak internals, and the request id in the body is
        enough to find the full traceback in the log.
        """
        logger.error("unhandled_exception", error=f"{type(exc).__name__}: {exc}", exc_info=True)
        return _error_response(request, "internal_error", "an unexpected error occurred", 500)

    app.include_router(admin_router)
    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(documents_router)
    return app


__all__ = [
    "REQUEST_ID_HEADER",
    "UNLIMITED_PATHS",
    "RateLimiter",
    "TokenBucket",
    "client_key",
    "create_app",
]

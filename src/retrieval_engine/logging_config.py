"""structlog configuration: JSON lines with a request id bound to every event.

The request id lives in a contextvar, so a log line emitted deep inside the retrieval
pipeline carries the id of the request that caused it without threading it through
every signature.
"""

from __future__ import annotations

import logging
import uuid
from typing import cast

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars
from structlog.typing import FilteringBoundLogger, Processor

REQUEST_ID_KEY = "request_id"

_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    """Configure structlog once, at process start.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR.
        json_logs: Emit JSON lines (production) or a readable console format (dev).
    """
    level_no = _LEVELS.get(level.upper(), logging.INFO)

    processors: list[Processor] = [
        merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # Keep uvicorn and library logs at the same level so nothing surprising is hidden.
    logging.basicConfig(level=level_no, format="%(message)s")


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a bound logger, optionally named after the calling module."""
    return cast(FilteringBoundLogger, structlog.get_logger(name))


def new_request_id() -> str:
    """Generate a request id for callers that did not supply ``X-Request-ID``."""
    return uuid.uuid4().hex


def bind_request_id(request_id: str) -> str:
    """Bind ``request_id`` to the logging context and return it."""
    bind_contextvars(**{REQUEST_ID_KEY: request_id})
    return request_id


def clear_request_context() -> None:
    """Clear all bound contextvars at the end of a request."""
    clear_contextvars()


__all__ = [
    "REQUEST_ID_KEY",
    "bind_request_id",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    "new_request_id",
]

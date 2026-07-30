"""Typed errors for the whole service.

Every error carries a stable machine-readable ``code`` and the HTTP status the API
should map it to, so the single exception handler in :mod:`retrieval_engine.api.app`
never needs a chain of isinstance checks.
"""

from __future__ import annotations


class RetrievalEngineError(Exception):
    """Base class for every error this service raises deliberately."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(RetrievalEngineError):
    """Settings are internally inconsistent, e.g. a dimension that cannot hold."""

    code = "configuration_error"
    http_status = 500


class UnsupportedFormatError(RetrievalEngineError):
    """A loader was handed a file extension nothing claims to read."""

    code = "unsupported_format"
    http_status = 415


class DocumentLoadError(RetrievalEngineError):
    """A file has a supported extension but could not be parsed."""

    code = "document_load_error"
    http_status = 422


class EmbeddingSpaceMismatchError(RetrievalEngineError):
    """Vectors from a different embedder or dimension were offered to a collection.

    Mixing embedding spaces silently destroys recall, and the damage is invisible
    until someone measures it, so this is a hard failure rather than a warning.
    """

    code = "embedding_space_mismatch"
    http_status = 409


class EmbeddingBackendError(RetrievalEngineError):
    """A remote embedding backend refused or failed a request.

    Distinct from :class:`EmbeddingSpaceMismatchError`: the vectors would have been
    fine, the service did not produce them.
    """

    code = "embedding_backend_error"
    http_status = 502


class StoreUnavailableError(RetrievalEngineError):
    """The vector store cannot be reached or has not been migrated."""

    code = "store_unavailable"
    http_status = 503


class DocumentNotFoundError(RetrievalEngineError):
    """A document id does not exist in the store."""

    code = "document_not_found"
    http_status = 404


class LLMUnavailableError(RetrievalEngineError):
    """The generation backend is unreachable or returned an unusable response.

    Callers are expected to catch this and fall back to extractive answering; the
    API must never surface it as a 5xx.
    """

    code = "llm_unavailable"
    http_status = 503


class RequestTooLargeError(RetrievalEngineError):
    """An upload exceeded the configured size limit."""

    code = "request_too_large"
    http_status = 413


class RateLimitExceededError(RetrievalEngineError):
    """The caller exhausted its token bucket."""

    code = "rate_limit_exceeded"
    http_status = 429


class GoldenSetValidationError(RetrievalEngineError):
    """A golden-set entry violates the substring or length contract."""

    code = "golden_set_invalid"
    http_status = 422


class JobNotFoundError(RetrievalEngineError):
    """An ingestion job id is unknown."""

    code = "job_not_found"
    http_status = 404


__all__ = [
    "ConfigurationError",
    "DocumentLoadError",
    "DocumentNotFoundError",
    "EmbeddingBackendError",
    "EmbeddingSpaceMismatchError",
    "GoldenSetValidationError",
    "JobNotFoundError",
    "LLMUnavailableError",
    "RateLimitExceededError",
    "RequestTooLargeError",
    "RetrievalEngineError",
    "StoreUnavailableError",
    "UnsupportedFormatError",
]

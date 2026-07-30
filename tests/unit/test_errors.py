"""Error hierarchy: the API's single exception handler reads ``code`` and ``http_status``.

These values are part of the public contract (clients switch on ``code``), so they are
asserted here rather than left to drift.
"""

from __future__ import annotations

import pytest

from retrieval_engine.errors import (
    ConfigurationError,
    DocumentLoadError,
    DocumentNotFoundError,
    EmbeddingSpaceMismatchError,
    GoldenSetValidationError,
    JobNotFoundError,
    LLMUnavailableError,
    RateLimitExceededError,
    RetrievalEngineError,
    StoreUnavailableError,
    UnsupportedFormatError,
)

ALL_ERRORS = [
    ConfigurationError,
    DocumentLoadError,
    DocumentNotFoundError,
    EmbeddingSpaceMismatchError,
    GoldenSetValidationError,
    JobNotFoundError,
    LLMUnavailableError,
    RateLimitExceededError,
    StoreUnavailableError,
    UnsupportedFormatError,
]


@pytest.mark.parametrize("error_class", ALL_ERRORS)
def test_every_error_is_catchable_as_the_base(error_class: type[RetrievalEngineError]) -> None:
    """One except clause in the API handler must catch everything we raise on purpose."""
    with pytest.raises(RetrievalEngineError):
        raise error_class("boom")


@pytest.mark.parametrize("error_class", ALL_ERRORS)
def test_message_is_preserved(error_class: type[RetrievalEngineError]) -> None:
    error = error_class("something specific")

    assert error.message == "something specific"
    assert str(error) == "something specific"


@pytest.mark.parametrize(
    ("error_class", "code", "status"),
    [
        (UnsupportedFormatError, "unsupported_format", 415),
        (DocumentLoadError, "document_load_error", 422),
        (EmbeddingSpaceMismatchError, "embedding_space_mismatch", 409),
        (StoreUnavailableError, "store_unavailable", 503),
        (DocumentNotFoundError, "document_not_found", 404),
        (LLMUnavailableError, "llm_unavailable", 503),
        (RateLimitExceededError, "rate_limit_exceeded", 429),
        (GoldenSetValidationError, "golden_set_invalid", 422),
        (JobNotFoundError, "job_not_found", 404),
        (ConfigurationError, "configuration_error", 500),
    ],
)
def test_code_and_status_mapping(
    error_class: type[RetrievalEngineError], code: str, status: int
) -> None:
    assert error_class.code == code
    assert error_class.http_status == status


def test_base_defaults_to_internal_error() -> None:
    assert RetrievalEngineError.code == "internal_error"
    assert RetrievalEngineError.http_status == 500


def test_error_codes_are_unique() -> None:
    """Clients switch on the code, so two errors sharing one would be ambiguous."""
    codes = [cls.code for cls in ALL_ERRORS]

    assert len(codes) == len(set(codes))

"""Logging: JSON output, level filtering, and request-id propagation via contextvars."""

from __future__ import annotations

import json

import pytest

from retrieval_engine.logging_config import (
    REQUEST_ID_KEY,
    bind_request_id,
    clear_request_context,
    configure_logging,
    get_logger,
    new_request_id,
)


@pytest.fixture(autouse=True)
def _clean_context() -> None:
    clear_request_context()


def test_json_logs_are_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=True)

    get_logger("test").info("ingest_finished", chunks=12)

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["event"] == "ingest_finished"
    assert payload["chunks"] == 12
    assert payload["level"] == "info"
    assert "timestamp" in payload


def test_request_id_is_attached_to_every_event(capsys: pytest.CaptureFixture[str]) -> None:
    """A log line emitted deep in the pipeline still carries the causing request id."""
    configure_logging("INFO", json_logs=True)
    request_id = bind_request_id("abc123")

    get_logger("test").warning("slow_stage", stage="rerank")

    payload = json.loads(capsys.readouterr().out.strip())
    assert request_id == "abc123"
    assert payload[REQUEST_ID_KEY] == "abc123"


def test_clearing_context_drops_the_request_id(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=True)
    bind_request_id("abc123")
    clear_request_context()

    get_logger("test").info("standalone")

    payload = json.loads(capsys.readouterr().out.strip())
    assert REQUEST_ID_KEY not in payload


def test_level_filtering_suppresses_lower_levels(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", json_logs=True)

    logger = get_logger("test")
    logger.info("should_not_appear")
    logger.error("should_appear")

    out = capsys.readouterr().out
    assert "should_not_appear" not in out
    assert "should_appear" in out


def test_unknown_level_falls_back_to_info(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("NOT_A_LEVEL", json_logs=True)

    get_logger("test").info("still_logged")

    assert "still_logged" in capsys.readouterr().out


def test_console_renderer_is_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG", json_logs=False)

    get_logger("test").debug("readable_line", stage="dense")

    out = capsys.readouterr().out
    assert "readable_line" in out
    assert not out.strip().startswith("{")


def test_request_ids_are_unique() -> None:
    assert new_request_id() != new_request_id()
    assert len(new_request_id()) == 32

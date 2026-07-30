"""Ingestion routes.

Two entry points rather than one, because a JSON body and a multipart upload cannot share a
FastAPI signature: ``POST /v1/ingest`` ingests a server-side directory, ``POST
/v1/ingest/upload`` ingests uploaded files.

A server-side path is only accepted if it resolves inside the configured data directory.
Without that check, a route that ingests "any directory path" is a way to read arbitrary
files off the host and get them back out through query answers.

Large batches run as jobs. The response returns immediately with an id, and the work
continues on a task the application keeps a reference to, because a bare ``create_task``
result can be garbage collected mid-run.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from retrieval_engine.api.deps import Services, ServicesDep
from retrieval_engine.errors import (
    DocumentLoadError,
    JobNotFoundError,
    RequestTooLargeError,
    RetrievalEngineError,
    UnsupportedFormatError,
)
from retrieval_engine.ingest.loaders import SUPPORTED_EXTENSIONS
from retrieval_engine.logging_config import get_logger
from retrieval_engine.models import (
    ErrorResponse,
    IngestJob,
    IngestRequest,
    IngestSummary,
    JobStatus,
    utc_now,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/ingest", tags=["ingest"])

# Literal status codes rather than the starlette constants: two of the ones needed here were
# renamed and now emit deprecation warnings, and the numbers themselves are not going to change.
HTTP_404_NOT_FOUND = 404
HTTP_413_TOO_LARGE = 413
HTTP_415_UNSUPPORTED = 415
HTTP_422_UNPROCESSABLE = 422

_BYTES_PER_MB = 1024 * 1024


def resolve_corpus_path(services: Services, raw: str | None) -> Path:
    """Resolve a requested path, refusing anything outside the data directory.

    A route that ingests an arbitrary server path is a file-read primitive: point it at a
    directory of secrets and the contents come back out through query answers. The
    containment check is the whole reason this function exists.
    """
    root = Path(services.settings.data_dir).resolve()
    if raw is None:
        return Path(services.settings.corpus_dir)
    candidate = Path(raw)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        msg = f"path {raw!r} is outside the configured data directory"
        raise DocumentLoadError(msg)
    return resolved


async def _run_job(services: Services, job: IngestJob, paths: list[Path]) -> None:
    """Execute one ingestion job, recording its outcome on the job record."""
    job.status = JobStatus.RUNNING
    job.updated_at = utc_now()
    try:
        job.summary = await services.ingest.ingest_paths(paths, progress=False)
        job.status = JobStatus.SUCCEEDED
    except RetrievalEngineError as exc:
        job.status = JobStatus.FAILED
        job.error = exc.message
        logger.warning("ingest_job_failed", job_id=job.job_id, error=exc.message)
    except Exception as exc:  # a background job must not die silently
        job.status = JobStatus.FAILED
        job.error = f"{type(exc).__name__}: {exc}"
        logger.warning("ingest_job_crashed", job_id=job.job_id, error=job.error)
    finally:
        job.updated_at = utc_now()


def _start_job(services: Services, paths: list[Path]) -> IngestJob:
    job = IngestJob(job_id=uuid.uuid4().hex, status=JobStatus.QUEUED)
    services.jobs[job.job_id] = job
    task = asyncio.create_task(_run_job(services, job, paths))
    # Hold a reference: a task with no strong reference can be collected mid-run.
    services.tasks.add(task)
    task.add_done_callback(services.tasks.discard)
    return job


@router.post(
    "",
    responses={
        HTTP_422_UNPROCESSABLE: {
            "model": ErrorResponse,
            "description": "Path outside the data directory, or unreadable",
        }
    },
    summary="Ingest a server-side directory",
)
async def ingest_directory(body: IngestRequest, services: ServicesDep) -> IngestSummary | IngestJob:
    """Ingest every loadable file under a directory inside the data directory."""
    from retrieval_engine.ingest.loaders import iter_source_files

    directory = resolve_corpus_path(services, body.path)
    paths = iter_source_files(directory)

    if body.async_job:
        return _start_job(services, paths)
    return await services.ingest.ingest_paths(paths, progress=False)


@router.post(
    "/upload",
    responses={
        HTTP_413_TOO_LARGE: {
            "model": ErrorResponse,
            "description": "Upload exceeds the configured limit",
        },
        HTTP_415_UNSUPPORTED: {
            "model": ErrorResponse,
            "description": "Unsupported file type",
        },
    },
    summary="Ingest uploaded files",
)
async def ingest_upload(
    services: ServicesDep,
    files: Annotated[list[UploadFile], File(description="Files to ingest.")],
) -> IngestSummary:
    """Ingest uploaded files.

    Uploads land in a temporary directory that is removed when the request finishes. They
    are never written into the corpus, so an upload cannot quietly become part of a
    reproducible eval corpus.
    """
    limit = services.settings.max_upload_mb * _BYTES_PER_MB
    with tempfile.TemporaryDirectory(prefix="re-upload-") as staging:
        directory = Path(staging)
        written: list[Path] = []
        for upload in files:
            name = Path(upload.filename or "unnamed").name
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                msg = f"unsupported file type for upload {name!r}"
                raise UnsupportedFormatError(msg)
            payload = await upload.read()
            if len(payload) > limit:
                msg = (
                    f"{name} is {len(payload) // _BYTES_PER_MB} MB, over the "
                    f"{services.settings.max_upload_mb} MB limit"
                )
                raise RequestTooLargeError(msg)
            target = directory / name
            target.write_bytes(payload)
            written.append(target)
        return await services.ingest.ingest_paths(written, progress=False)


@router.get(
    "/{job_id}",
    response_model=IngestJob,
    responses={HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Unknown job"}},
    summary="Ingestion job status",
)
async def job_status(job_id: str, services: ServicesDep) -> IngestJob:
    """Status of an asynchronous ingestion job."""
    job = services.jobs.get(job_id)
    if job is None:
        msg = f"no ingestion job with id {job_id!r}"
        raise JobNotFoundError(msg)
    return job


__all__ = ["resolve_corpus_path", "router"]

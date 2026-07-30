# Multi-stage build. The layer order is the whole point of this file.
#
# Dependencies are installed before the source is copied, and the model weights are baked
# before that again, because those two layers are the slow ones (torch is hundreds of
# megabytes, the two models together are over a gigabyte) and they change least often.
# Copying source first would invalidate both on every code edit and turn a ten second
# rebuild into a ten minute one.

FROM python:3.12-slim AS builder

# Pinned rather than :latest, so a rebuild six months from now resolves the same way.
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/hf

WORKDIR /app

# Only the files uv needs to resolve. README.md is here because pyproject declares it as the
# project readme and the build backend reads it.
COPY pyproject.toml uv.lock README.md ./

# --frozen makes uv.lock authoritative: the build fails rather than silently resolving
# something different from what was tested. --no-dev keeps pytest and friends out of the image.
RUN uv sync --frozen --no-install-project --no-dev

# Bake the weights. Without this the first request in a fresh container pays a 1.5 GB
# download, which turns a cold start into a timeout behind any sensible readiness probe.
RUN /app/.venv/bin/python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base'); \
print('weights cached')"

COPY src ./src
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

ENV HF_HOME=/opt/hf \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The weights are already in the image, so the runtime never needs the network for them.
    # If a model id is ever changed without rebuilding, this fails loudly instead of quietly
    # downloading a different model in production.
    HF_HUB_OFFLINE=1

# Non-root. A retrieval service ingests files it was handed, so it should not be able to
# write anywhere it was not given.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /opt/hf /opt/hf
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser migrations ./migrations
COPY --chown=appuser:appuser scripts ./scripts

USER appuser
EXPOSE 8000

# Hits the app's own liveness route rather than probing the port, because the port accepts
# connections well before the application is able to answer.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "retrieval_engine.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]

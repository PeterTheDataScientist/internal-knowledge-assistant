# Two stages so the runtime image does not carry the build toolchain.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt requirements-ml.txt ./
# The image carries the neural embedder, because an image is built once and the download
# cost is paid by the build rather than by the person running it.
RUN pip install --prefix=/install -r requirements-ml.txt


FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/app/.cache/huggingface

# The service runs as a non-root user. A container that can write to its own filesystem
# is a lateral movement step, and there is no reason this process needs it.
RUN useradd --create-home --uid 10001 app

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=app:app app ./app
COPY --chown=app:app data ./data
COPY --chown=app:app eval ./eval
COPY --chown=app:app tests ./tests
COPY --chown=app:app pyproject.toml openapi.yaml ./

USER app

EXPOSE 8000

# The image is useful with no network access: with no API key the service starts on the
# offline provider rather than failing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

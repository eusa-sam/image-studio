FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    IMAGE_STUDIO_CONTAINER=1 \
    PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-cache
COPY backend/ backend/
COPY scripts/ scripts/
RUN mkdir -p models data outputs
EXPOSE 8765
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/image-runtime', timeout=3)"
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8765", "--no-access-log"]

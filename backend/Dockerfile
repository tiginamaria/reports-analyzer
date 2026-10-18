FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Hosted embeddings by default, so neither torch nor sentence-transformers goes in the image.
# To run local models in Docker: --build-arg LOCAL_MODELS=1 (adds about 2 GB).
ARG LOCAL_MODELS=0
RUN if [ "$LOCAL_MODELS" = "1" ]; then uv sync --frozen --no-dev --no-install-project; \
    else uv sync --frozen --no-dev --no-group local-models --no-install-project; fi
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "uv run alembic upgrade head && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000"]

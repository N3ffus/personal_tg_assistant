# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

FROM python:3.13.9-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM python:3.13.9-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

RUN useradd --create-home --uid 10001 app

WORKDIR /app

EXPOSE 8080

COPY --from=builder --chown=app:app /app /app

RUN mkdir -p /app/data && chown app:app /app/data

USER app

CMD ["python", "-m", "src.main"]

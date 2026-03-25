# Stage 1: install dependencies
# Official uv image includes both uv and Python 3.13
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

# Optional: private PyPI index for internal packages (e.g. torchtextclassifiers)
# Pass --build-arg INSEE_NEXUS_URL=https://nexus.example.com/repository/pypi/simple
ARG INSEE_NEXUS_URL=""

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy only dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install all dependencies into a virtual environment.
# The uv cache is mounted to speed up rebuilds without polluting the image layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$INSEE_NEXUS_URL" ]; then \
      UV_EXTRA_INDEX_URL="$INSEE_NEXUS_URL" uv sync --frozen --no-dev --no-install-project; \
    else \
      uv sync --frozen --no-dev --no-install-project; \
    fi

# Stage 2: runtime image (no uv needed at runtime)
FROM python:3.13-slim

WORKDIR /app

# Copy only the virtualenv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY main.py ./
COPY src/ ./src/
COPY data/ ./data/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "main.py"]

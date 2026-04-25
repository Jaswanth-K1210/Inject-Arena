# ---------------------------------------------------------------------------
# Stage 1 — build the React/Vite frontend
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ---------------------------------------------------------------------------
# Stage 2 — Python runtime serving FastAPI + the built frontend
# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the source tree (excluding node_modules via .dockerignore) and install
# CPU-only Python deps. [gpu] extras are intentionally skipped so the image
# stays small enough for the free Spaces tier.
COPY . .
RUN pip install --no-cache-dir -e ".[demo]"

# Copy the built frontend from stage 1 into frontend/dist/.
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

# Replay-mode defaults for HF Spaces: traces baked in, no GPU required.
ENV USE_STUB_DEFENSES=true
ENV INJECTARENA_MODE=replay
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860"]

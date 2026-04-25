FROM python:3.11-slim

WORKDIR /app

# Base OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# CPU-only deps. Excludes [gpu] so the image stays small enough for free Spaces.
RUN pip install --no-cache-dir -e ".[demo]"

# HF Spaces / replay-mode defaults: no GPU, no model downloads, traces baked in.
ENV USE_STUB_DEFENSES=true
ENV INJECTARENA_MODE=replay
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces uses port 7860.
EXPOSE 7860

CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860"]

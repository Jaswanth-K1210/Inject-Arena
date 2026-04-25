FROM python:3.11-slim

WORKDIR /app

# Install base OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install CPU-only deps (no GPU in HF Spaces Docker)
RUN pip install --no-cache-dir -e ".[demo]"

# Stub defenses: no GPU, no model downloads in Space
ENV USE_STUB_DEFENSES=true

# HuggingFace Spaces uses port 7860
EXPOSE 7860

CMD ["python", "demo/gradio_app.py"]

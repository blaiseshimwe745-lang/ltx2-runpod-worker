FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VOLUME_PATH=/runpod-volume \
    PATH="/root/.local/bin:/opt/LTX-2/.venv/bin:${PATH}"

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates ffmpeg python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- uv (fast Python installer used by Lightricks LTX-2) ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ---- clone LTX-2 monorepo and install via uv ----
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /opt/LTX-2
WORKDIR /opt/LTX-2
RUN uv sync --frozen

# ---- our extras into the same venv ----
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN /opt/LTX-2/.venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# ---- handler + downloader ----
COPY download_models.py /app/download_models.py
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]

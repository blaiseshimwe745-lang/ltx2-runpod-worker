FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VOLUME_PATH=/runpod-volume \
    PATH="/root/.local/bin:/opt/LTX-2/.venv/bin:${PATH}"

# ---- system deps (Ubuntu 24.04 ships with Python 3.12 natively) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates ffmpeg \
        python3 python3-venv python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# ---- uv (fast Python installer used by Lightricks LTX-2) ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ---- clone LTX-2 monorepo and install via uv ----
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /opt/LTX-2
WORKDIR /opt/LTX-2
RUN /root/.local/bin/uv sync --frozen || /root/.local/bin/uv sync

# ---- our extras into the same venv ----
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN /root/.local/bin/uv pip install --python /opt/LTX-2/.venv/bin/python --no-cache-dir -r /app/requirements.txt

# ---- handler + downloader ----
COPY download_models.py /app/download_models.py
COPY handler.py /app/handler.py

CMD ["/opt/LTX-2/.venv/bin/python", "-u", "/app/handler.py"]

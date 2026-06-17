FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    pillow \
    decord \
    imageio \
    opencv-python \
    huggingface_hub

COPY . /app
WORKDIR /app

CMD ["python", "-u", "/app/handler.py"]

# Multi-stage build for Raspberry Pi 5 + Hailo
# Base image: Raspberry Pi OS Bookworm (arm64)

FROM python:3.11-slim-bookworm AS base

WORKDIR /app

# System deps: OpenCV runtime, camera libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python deps (no Hailo here — installed via host SDK mount or .whl)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: Hailo SDK (.whl) must be installed on the Pi host and bind-mounted,
# or pre-installed into this image if you have the .whl file.
# Uncomment and adjust path if you have it locally:
# COPY hailo_platform-4.x.x-py3-none-linux_aarch64.whl .
# RUN pip install hailo_platform-4.x.x-py3-none-linux_aarch64.whl

COPY . .

EXPOSE 5000

CMD ["python", "main.py", "--config", "config/settings.yaml"]

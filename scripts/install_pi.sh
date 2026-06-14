#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# scripts/install_pi.sh
# One-shot setup for Raspberry Pi Zero 2W running Pi OS Lite 64-bit
# Usage: bash scripts/install_pi.sh
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

echo "════════════════════════════════════════"
echo "  Pi-Detect — Raspberry Pi Setup"
echo "════════════════════════════════════════"

# Fix /tmp for large package installs
echo "[0/6] Fixing /tmp permissions..."
sudo chmod 1777 /tmp
mkdir -p ~/tmp
sudo mount --bind ~/tmp /tmp 2>/dev/null || true

# System update
echo "[1/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# System dependencies
echo "[2/6] Installing system dependencies..."
sudo apt-get install -y -qq \
  python3-pip \
  python3-picamera2 \
  libcamera-apps \
  libopencv-dev \
  python3-opencv \
  git curl tmux \
  libatlas-base-dev

# Enable camera
echo "[3/6] Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Python base packages
echo "[4/6] Installing Python packages..."
pip install flask opencv-python-headless python-dotenv numpy matplotlib \
  psutil ultralytics-thop --break-system-packages --cache-dir ~/pip-cache

# PyTorch CPU (large download — be patient)
echo "[5/6] Installing PyTorch CPU (150MB — ~5 mins)..."
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu \
  --break-system-packages \
  --cache-dir ~/pip-cache

# Ultralytics
echo "[6/6] Installing Ultralytics YOLOv8..."
pip install ultralytics --no-deps --break-system-packages

# Project setup
mkdir -p models logs

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env — edit it now: nano .env"
fi

echo ""
echo "✓ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. nano .env  (set USE_PICAMERA=true)"
echo "  2. bash scripts/start.sh"

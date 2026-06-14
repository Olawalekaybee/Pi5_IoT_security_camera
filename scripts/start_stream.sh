#!/usr/bin/env bash
# scripts/start_stream.sh
# Starts the Pi-Detect server (optionally with ngrok tunnel)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .env 2>/dev/null || true

echo "════════════════════════════════════════"
echo "  Pi-Detect — Starting Server"
echo "════════════════════════════════════════"
echo "  Resolution : ${CAMERA_WIDTH:-640}x${CAMERA_HEIGHT:-480}"
echo "  Model      : ${MODEL_NAME:-yolov8n.pt}"
echo "  Skip frames: ${DETECTION_SKIP_FRAMES:-3}"
echo "════════════════════════════════════════"

mkdir -p logs

# Start ngrok tunnel in background if token is set
if [ -n "${NGROK_AUTHTOKEN:-}" ] && [ "$NGROK_AUTHTOKEN" != "your_ngrok_auth_token_here" ]; then
  echo "Starting ngrok tunnel…"
  bash scripts/tunnel.sh &
  sleep 2
fi

# Start Flask server
echo "Starting Pi-Detect server on port ${PORT:-5000}…"
python3 main.py --host "${HOST:-0.0.0.0}" --port "${PORT:-5000}"

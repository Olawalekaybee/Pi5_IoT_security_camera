#!/usr/bin/env bash
# scripts/start.sh — Start Pi-Detect server
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .env 2>/dev/null || true

echo "════════════════════════════════════════"
echo "  Pi-Detect — Starting"
echo "  Resolution : ${CAMERA_WIDTH:-320}x${CAMERA_HEIGHT:-240}"
echo "  Model      : ${MODEL_NAME:-yolov8n.pt}"
echo "════════════════════════════════════════"

mkdir -p logs
sudo mount --bind ~/tmp /tmp 2>/dev/null || true

python3 main.py --host "${HOST:-0.0.0.0}" --port "${PORT:-5000}"

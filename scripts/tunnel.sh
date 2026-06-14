#!/usr/bin/env bash
# scripts/tunnel.sh
# Opens an ngrok HTTPS tunnel to the Flask server.
# Prints the public URL so you can open it on any device over 4G.
set -euo pipefail

source "$(dirname "$0")/../.env" 2>/dev/null || true

PORT="${PORT:-5000}"

# Authenticate ngrok
ngrok authtoken "$NGROK_AUTHTOKEN" 2>/dev/null

# Start tunnel and print URL
echo "Opening ngrok tunnel on port $PORT…"
ngrok http "$PORT" --log=stdout --log-level=info &
NGROK_PID=$!

# Wait for tunnel to come up, then print the URL
sleep 4
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['tunnels'][0]['public_url'])" 2>/dev/null || echo "")

if [ -n "$PUBLIC_URL" ]; then
  echo ""
  echo "╔══════════════════════════════════════╗"
  echo "  Public URL (share this):"
  echo "  $PUBLIC_URL"
  echo "╚══════════════════════════════════════╝"
  echo ""
else
  echo "ngrok started — check http://localhost:4040 for your public URL"
fi

wait $NGROK_PID

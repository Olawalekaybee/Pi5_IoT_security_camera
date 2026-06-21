"""
Flask dashboard: serves a live-updating page with the annotated video
feed (MJPEG) alongside a recent-events table updated via Server-Sent
Events (SSE), so the browser updates in real time without polling.
"""

from __future__ import annotations
import json
import time
import logging
from flask import Flask, Response, render_template_string, jsonify

logger = logging.getLogger(__name__)

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Edge AI Security Monitor</title>
<style>
  :root {
    --bg: #0b0c0f; --panel: #15171c; --panel-2: #1b1e25; --border: #262a33;
    --text: #e8e9ec; --text-dim: #8b8f99; --accent: #5b8cff;
    --ok: #45c98a; --warn: #ff6b6b;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); margin: 0; padding: 1.5rem 2rem;
  }
  header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 1.25rem; }
  header h1 { font-size: 1.25rem; font-weight: 600; margin: 0; letter-spacing: -0.01em; }
  .live-pill {
    display: inline-flex; align-items: center; gap: 6px; font-size: 0.75rem;
    color: var(--ok); background: rgba(69,201,138,0.12); border: 1px solid rgba(69,201,138,0.3);
    border-radius: 20px; padding: 3px 10px;
  }
  .live-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ok); animation: pulse 1.6s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

  .layout { display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(280px, 1fr); gap: 1.25rem; align-items: start; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
  .panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.7rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.8rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .video-wrap { background: #000; display: flex; align-items: center; justify-content: center; min-height: 280px; }
  .video-wrap img { width: 100%; display: block; }
  .video-offline { color: var(--text-dim); font-size: 0.85rem; padding: 2rem; text-align: center; }

  .legend { display: flex; gap: 1rem; padding: 0.6rem 1rem; font-size: 0.75rem; color: var(--text-dim); }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.6rem; margin-bottom: 1.25rem; }
  .stat { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 0.8rem 1rem; }
  .stat .label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.03em; }
  .stat .value { font-size: 1.4rem; font-weight: 600; margin-top: 2px; }

  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th, td { text-align: left; padding: 0.55rem 1rem; border-bottom: 1px solid var(--border); }
  th { color: var(--text-dim); font-weight: 500; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; }
  tbody tr:hover { background: var(--panel-2); }
  .alerted { color: var(--warn); font-weight: 600; }
  .ok-status { color: var(--ok); }
  .events-scroll { max-height: 480px; overflow-y: auto; }
</style>
</head>
<body>
  <header>
    <h1>Edge AI Security Monitor</h1>
    <span class="live-pill"><span class="live-dot"></span>LIVE</span>
  </header>

  <div class="stats" id="stats"></div>

  <div class="layout">
    <div class="panel">
      <div class="panel-header"><span>Camera 1 — Live Feed</span><span id="feed-status"></span></div>
      <div class="video-wrap">
        <img id="video-feed" src="/video_feed" alt="Live camera feed"
             onerror="this.style.display='none'; document.getElementById('video-offline').style.display='block';">
        <div class="video-offline" id="video-offline" style="display:none;">Camera feed unavailable</div>
      </div>
      <div class="legend">
        <span><span class="swatch" style="background:#50c878;"></span>Recognized</span>
        <span><span class="swatch" style="background:#e63c3c;"></span>Unknown / Alert</span>
        <span><span class="swatch" style="background:#c8a03c;"></span>Zone boundary</span>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header"><span>Recent Events</span></div>
      <div class="events-scroll">
        <table id="events">
          <thead><tr><th>Time</th><th>Zone</th><th>Conf</th><th>Person</th><th>Status</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

<script>
const evtSource = new EventSource("/stream");
evtSource.onmessage = function(e) {
  const data = JSON.parse(e.data);
  document.getElementById("stats").innerHTML = `
    <div class="stat"><div class="label">Total events</div><div class="value">${data.stats.total_events}</div></div>
    <div class="stat"><div class="label">Total alerts</div><div class="value">${data.stats.total_alerts}</div></div>
    <div class="stat"><div class="label">Last 24h</div><div class="value">${data.stats.events_24h}</div></div>
  `;
  const tbody = document.querySelector("#events tbody");
  tbody.innerHTML = data.events.map(ev => `
    <tr>
      <td>${new Date(ev.timestamp * 1000).toLocaleTimeString()}</td>
      <td>${ev.zone}</td>
      <td>${(ev.detection_confidence * 100).toFixed(0)}%</td>
      <td>${ev.person_id || "unknown"}</td>
      <td class="${ev.alerted ? 'alerted' : 'ok-status'}">${ev.alerted ? "ALERTED" : "logged"}</td>
    </tr>
  `).join("");
};
</script>
</body>
</html>
"""


class DashboardServer:
    def __init__(self, db, settings, pipeline=None):
        """
        pipeline is optional so the dashboard can still run standalone
        (e.g. for UI development) without a live detection pipeline;
        when provided, its get_latest_jpeg() feeds the /video_feed route.
        """
        self.db = db
        self.settings = settings
        self.pipeline = pipeline
        self.app = Flask(__name__)
        self._register_routes()

    def _register_routes(self):
        app = self.app

        @app.route("/")
        def index():
            return render_template_string(PAGE_TEMPLATE)

        @app.route("/api/events")
        def api_events():
            events = self.db.get_recent_events(limit=self.settings.dashboard.max_events_page)
            return jsonify(events)

        @app.route("/api/stats")
        def api_stats():
            return jsonify(self.db.get_stats())

        @app.route("/stream")
        def stream():
            def event_stream():
                while True:
                    payload = {
                        "events": self.db.get_recent_events(
                            limit=self.settings.dashboard.max_events_page
                        ),
                        "stats": self.db.get_stats(),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    time.sleep(1.5)

            return Response(event_stream(), mimetype="text/event-stream")

        @app.route("/video_feed")
        def video_feed():
            if self.pipeline is None:
                return Response(status=503)

            def mjpeg_stream():
                boundary = b"--frame"
                while True:
                    jpeg = self.pipeline.get_latest_jpeg()
                    if jpeg is not None:
                        yield (
                            boundary + b"\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                            + jpeg + b"\r\n"
                        )
                    time.sleep(1.0 / max(self.settings.dashboard.video_feed_fps, 1))

            return Response(
                mjpeg_stream(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def run(self):
        self.app.run(
            host=self.settings.dashboard.host,
            port=self.settings.dashboard.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
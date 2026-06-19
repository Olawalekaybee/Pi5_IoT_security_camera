"""
Flask dashboard: serves a live-updating page of recent detection
events using Server-Sent Events (SSE), so the browser updates in
real time without polling.
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
<title>Edge AI Security Monitor</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 0; padding: 2rem; }
  h1 { font-size: 1.4rem; font-weight: 500; }
  .stats { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
  .stat { background: #1c1c1c; border-radius: 8px; padding: 1rem 1.5rem; }
  .stat .label { font-size: 0.75rem; color: #999; text-transform: uppercase; }
  .stat .value { font-size: 1.5rem; font-weight: 500; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #2a2a2a; font-size: 0.85rem; }
  th { color: #999; font-weight: 500; }
  .alerted { color: #e2807a; font-weight: 500; }
  .ok { color: #7ad9a5; }
</style>
</head>
<body>
  <h1>Edge AI Security Monitor</h1>
  <div class="stats" id="stats"></div>
  <table id="events">
    <thead><tr><th>Time</th><th>Zone</th><th>Confidence</th><th>Person</th><th>Status</th></tr></thead>
    <tbody></tbody>
  </table>

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
      <td class="${ev.alerted ? 'alerted' : 'ok'}">${ev.alerted ? "ALERTED" : "logged"}</td>
    </tr>
  `).join("");
};
</script>
</body>
</html>
"""


class DashboardServer:
    def __init__(self, db, settings):
        self.db = db
        self.settings = settings
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

    def run(self):
        self.app.run(
            host=self.settings.dashboard.host,
            port=self.settings.dashboard.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

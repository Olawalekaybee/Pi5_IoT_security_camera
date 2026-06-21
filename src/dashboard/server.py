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
  .spoof-status { color: #ffa500; font-weight: 600; }
  .events-scroll { max-height: 480px; overflow-y: auto; }

  .enroll-body { padding: 1rem; }
  .enroll-hint { font-size: 0.8rem; color: var(--text-dim); margin: 0 0 0.9rem; line-height: 1.5; }
  .enroll-controls { display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.8rem; }
  .enroll-controls input[type="text"] {
    background: var(--panel-2); border: 1px solid var(--border); color: var(--text);
    border-radius: 8px; padding: 0.5rem 0.75rem; font-size: 0.85rem; min-width: 180px;
  }
  .enroll-controls button {
    background: var(--accent); color: white; border: none; border-radius: 8px;
    padding: 0.5rem 0.9rem; font-size: 0.82rem; cursor: pointer; font-weight: 500;
  }
  .enroll-controls button:disabled { background: #3a3d46; cursor: not-allowed; opacity: 0.6; }
  #btn-clear { background: var(--panel-2); border: 1px solid var(--border); }
  .thumbs { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.6rem; min-height: 0; }
  .thumb { width: 64px; height: 64px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); }
  .enroll-status { font-size: 0.8rem; color: var(--text-dim); min-height: 1.2em; }
  .enroll-status.error { color: var(--warn); }
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
        <span><span class="swatch" style="background:#ffa500;"></span>Spoof suspected</span>
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

  <div class="panel" style="margin-top: 1.25rem;">
    <div class="panel-header"><span>Enroll Known Person</span></div>
    <div class="enroll-body">
      <p class="enroll-hint">
        Stand in frame, then capture 3-5 photos from different angles/distances for a robust match.
        Each capture grabs whoever is currently detected in the live feed.
      </p>
      <div class="enroll-controls">
        <input type="text" id="enroll-name" placeholder="Person's name" maxlength="40">
        <button id="btn-capture" onclick="captureForEnroll()">Capture Photo</button>
        <button id="btn-save" onclick="saveEnrollment()" disabled>Save Enrollment (0)</button>
        <button id="btn-clear" onclick="clearEnrollment()">Clear</button>
      </div>
      <div class="thumbs" id="enroll-thumbs"></div>
      <div class="enroll-status" id="enroll-status"></div>
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
  tbody.innerHTML = data.events.map(ev => {
    let statusClass = "ok-status";
    let statusText = "logged";
    if (ev.liveness_static === 1) {
      statusClass = "spoof-status";
      statusText = "SPOOF?";
    } else if (ev.alerted) {
      statusClass = "alerted";
      statusText = "ALERTED";
    }
    return `
    <tr>
      <td>${new Date(ev.timestamp * 1000).toLocaleTimeString()}</td>
      <td>${ev.zone}</td>
      <td>${(ev.detection_confidence * 100).toFixed(0)}%</td>
      <td>${ev.person_id || "unknown"}</td>
      <td class="${statusClass}">${statusText}</td>
    </tr>
  `;
  }).join("");
};

// --- Enrollment workflow ---
let stagedTokens = [];

function setEnrollStatus(msg, isError) {
  const el = document.getElementById("enroll-status");
  el.textContent = msg;
  el.className = "enroll-status" + (isError ? " error" : "");
}

function updateSaveButton() {
  const btn = document.getElementById("btn-save");
  btn.textContent = `Save Enrollment (${stagedTokens.length})`;
  btn.disabled = stagedTokens.length === 0;
}

async function captureForEnroll() {
  const btn = document.getElementById("btn-capture");
  btn.disabled = true;
  setEnrollStatus("Capturing...", false);
  try {
    const res = await fetch("/api/enroll/capture", { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      setEnrollStatus(data.error || "Capture failed", true);
      return;
    }
    stagedTokens.push(data.token);
    const thumbs = document.getElementById("enroll-thumbs");
    const img = document.createElement("img");
    img.src = data.thumbnail;
    img.className = "thumb";
    thumbs.appendChild(img);
    updateSaveButton();
    setEnrollStatus(`Captured ${stagedTokens.length} photo(s).`, false);
  } catch (err) {
    setEnrollStatus("Capture failed: " + err, true);
  } finally {
    btn.disabled = false;
  }
}

async function saveEnrollment() {
  const name = document.getElementById("enroll-name").value.trim();
  if (!name) {
    setEnrollStatus("Enter a name first.", true);
    return;
  }
  if (stagedTokens.length === 0) {
    setEnrollStatus("Capture at least one photo first.", true);
    return;
  }
  setEnrollStatus("Saving...", false);
  try {
    const res = await fetch("/api/enroll/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name, tokens: stagedTokens }),
    });
    const data = await res.json();
    if (!res.ok) {
      setEnrollStatus(data.error || "Save failed", true);
      return;
    }
    setEnrollStatus(`Enrolled '${data.name}' from ${data.photos_used} photo(s).`, false);
    clearEnrollment();
  } catch (err) {
    setEnrollStatus("Save failed: " + err, true);
  }
}

function clearEnrollment() {
  stagedTokens = [];
  document.getElementById("enroll-thumbs").innerHTML = "";
  updateSaveButton();
}
</script>
</body>
</html>
"""


class DashboardServer:
    def __init__(self, db, settings, pipeline=None, identifier=None):
        """
        pipeline is optional so the dashboard can still run standalone
        (e.g. for UI development) without a live detection pipeline;
        when provided, its get_latest_jpeg() feeds the /video_feed route.
        identifier, when provided, enables the in-dashboard enrollment
        flow (capturing fresh known-person photos from the live feed).
        """
        self.db = db
        self.settings = settings
        self.pipeline = pipeline
        self.identifier = identifier
        self.app = Flask(__name__)
        self._staged_crops = {}  # token -> numpy crop, pending enrollment save
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

        @app.route("/api/enroll/capture", methods=["POST"])
        def enroll_capture():
            """
            Grabs the current best person crop from the live feed and
            stages it in memory (keyed by name) for later saving. Returns
            a small JPEG thumbnail as base64 so the dashboard can show a
            preview before committing.
            """
            import cv2, base64

            if self.pipeline is None or self.identifier is None:
                return jsonify({"error": "capture not available"}), 503

            crop = self.pipeline.capture_person_crop()
            if crop is None:
                return jsonify({"error": "no person currently in frame"}), 404

            ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return jsonify({"error": "encode failed"}), 500

            token = f"capture_{int(time.time() * 1000)}"
            self._staged_crops[token] = crop

            thumb_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
            return jsonify({"token": token, "thumbnail": f"data:image/jpeg;base64,{thumb_b64}"})

        @app.route("/api/enroll/save", methods=["POST"])
        def enroll_save():
            """
            Enrolls the staged crops (by token, captured via
            /api/enroll/capture) under the given name, averaging their
            embeddings for a more robust reference than a single photo.
            """
            from flask import request

            if self.identifier is None:
                return jsonify({"error": "enrollment not available"}), 503

            data = request.get_json(silent=True) or {}
            name = (data.get("name") or "").strip()
            tokens = data.get("tokens") or []

            if not name:
                return jsonify({"error": "name is required"}), 400
            if not tokens:
                return jsonify({"error": "no captured photos to save"}), 400

            saved = 0
            for i, token in enumerate(tokens):
                crop = self._staged_crops.pop(token, None)
                if crop is None:
                    continue
                self.identifier.enroll(name, crop, average_with_existing=(saved > 0))
                saved += 1

            if saved == 0:
                return jsonify({"error": "none of the captured photos were found (expired?)"}), 400

            logger.info(f"Enrolled '{name}' from {saved} dashboard-captured photo(s)")
            return jsonify({"status": "ok", "name": name, "photos_used": saved})

    def run(self):
        self.app.run(
            host=self.settings.dashboard.host,
            port=self.settings.dashboard.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0c10; --panel: #111419; --panel-2: #161a21; --border: #232830;
    --text: #e4e7ec; --text-dim: #6b7280; --text-mid: #9aa1ad;
    --cyan: #3fd9e8; --cyan-dim: rgba(63,217,232,0.12);
    --ok: #4ade80; --warn: #f87171; --spoof: #fbbf24;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; overflow: hidden; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; flex-direction: column; height: 100vh;
  }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-track { background: transparent; }

  .mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }

  header {
    flex: 0 0 auto; display: flex; align-items: center; justify-content: space-between;
    padding: 0.7rem 1.25rem; border-bottom: 1px solid var(--border); background: var(--panel);
  }
  .brand { display: flex; align-items: center; gap: 0.6rem; }
  .brand-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }
  header h1 { font-size: 0.95rem; font-weight: 600; margin: 0; letter-spacing: 0.01em; }
  .header-right { display: flex; align-items: center; gap: 0.9rem; }
  .live-pill {
    display: inline-flex; align-items: center; gap: 6px; font-size: 0.68rem;
    color: var(--ok); background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.28);
    border-radius: 20px; padding: 3px 9px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.04em;
  }
  .live-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--ok); animation: pulse 1.6s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  .stat-strip { display: flex; gap: 1.4rem; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }
  .stat-strip .s-item { display: flex; align-items: baseline; gap: 0.4rem; }
  .stat-strip .s-label { color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.04em; }
  .stat-strip .s-value { color: var(--text); font-size: 0.95rem; font-weight: 600; }

  main { flex: 1 1 auto; height: 0; display: flex; gap: 1px; background: var(--border); overflow: hidden; }

  .video-col { flex: 1 1 auto; min-width: 0; background: var(--bg); display: flex; flex-direction: column; min-height: 0; }
  .video-header {
    flex: 0 0 auto; padding: 0.55rem 1rem; font-size: 0.7rem; color: var(--text-mid);
    text-transform: uppercase; letter-spacing: 0.05em; display: flex; justify-content: space-between;
    border-bottom: 1px solid var(--border); position: relative; overflow: hidden;
  }
  .video-header::after {
    content: ''; position: absolute; left: -40%; top: 0; bottom: 0; width: 40%;
    background: linear-gradient(90deg, transparent, var(--cyan-dim), transparent);
    animation: scan 4.5s linear infinite;
  }
  @keyframes scan { 0% { left: -40%; } 100% { left: 100%; } }
  .video-wrap { flex: 1 1 auto; background: #000; display: flex; align-items: center; justify-content: center; min-height: 0; position: relative; }
  .video-wrap img { max-width: 100%; max-height: 100%; display: block; }
  .video-offline { color: var(--text-dim); font-size: 0.85rem; }

  .legend { flex: 0 0 auto; display: flex; gap: 1.1rem; padding: 0.55rem 1rem; font-size: 0.7rem; color: var(--text-dim); border-top: 1px solid var(--border); }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .swatch { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }

  .side-col { flex: 0 0 360px; width: 360px; background: var(--bg); display: flex; flex-direction: column; min-height: 0; }

  .events-panel { flex: 1 1 auto; display: flex; flex-direction: column; min-height: 0; border-bottom: 1px solid var(--border); }
  .panel-title {
    flex: 0 0 auto; padding: 0.6rem 1rem; font-size: 0.7rem; color: var(--text-mid);
    text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border);
  }
  .events-scroll { flex: 1 1 auto; overflow-y: auto; min-height: 0; }
  table { width: 100%; border-collapse: collapse; font-size: 0.74rem; }
  th, td { text-align: left; padding: 0.4rem 0.8rem; font-family: 'JetBrains Mono', monospace; }
  th { color: var(--text-dim); font-weight: 500; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.04em;
       position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--border); padding-top: 0.5rem; padding-bottom: 0.5rem; }
  tbody tr { border-bottom: 1px solid rgba(35,40,48,0.6); }
  tbody tr:hover { background: var(--panel-2); }
  .alerted { color: var(--warn); font-weight: 600; }
  .ok-status { color: var(--ok); }
  .spoof-status { color: var(--spoof); font-weight: 600; }

  .enroll-panel { flex: 0 0 auto; max-height: 38%; display: flex; flex-direction: column; }
  .enroll-body { padding: 0.75rem 1rem; overflow-y: auto; }
  .enroll-hint { font-size: 0.7rem; color: var(--text-dim); margin: 0 0 0.65rem; line-height: 1.45; }
  .enroll-controls { display: flex; gap: 0.45rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.6rem; }
  .enroll-controls input[type="text"] {
    background: var(--panel-2); border: 1px solid var(--border); color: var(--text);
    border-radius: 6px; padding: 0.4rem 0.6rem; font-size: 0.78rem; flex: 1 1 140px; min-width: 0;
    font-family: inherit;
  }
  .enroll-controls input[type="text"]:focus { outline: none; border-color: var(--cyan); }
  .enroll-controls button {
    background: var(--cyan); color: #06222a; border: none; border-radius: 6px;
    padding: 0.4rem 0.7rem; font-size: 0.74rem; cursor: pointer; font-weight: 600;
    font-family: 'Inter', sans-serif; transition: opacity 0.15s;
  }
  .enroll-controls button:hover { opacity: 0.85; }
  .enroll-controls button:disabled { background: #2a2f38; color: var(--text-dim); cursor: not-allowed; opacity: 0.7; }
  #btn-clear { background: var(--panel-2); border: 1px solid var(--border); color: var(--text-mid); }
  .thumbs { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.45rem; }
  .thumb { width: 44px; height: 44px; object-fit: cover; border-radius: 5px; border: 1px solid var(--border); }
  .enroll-status { font-size: 0.72rem; color: var(--text-dim); min-height: 1.1em; font-family: 'JetBrains Mono', monospace; }
  .enroll-status.error { color: var(--warn); }
</style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="brand-dot"></span>
      <h1>EDGE AI SECURITY MONITOR</h1>
    </div>
    <div class="header-right">
      <div class="stat-strip" id="stats"></div>
      <span class="live-pill"><span class="live-dot"></span>LIVE</span>
    </div>
  </header>

  <main>
    <div class="video-col">
      <div class="video-header">
        <span>CAM 01 // LIVE FEED</span>
        <span id="feed-status" class="mono"></span>
      </div>
      <div class="video-wrap">
        <img id="video-feed" src="/video_feed" alt="Live camera feed"
             onerror="this.style.display='none'; document.getElementById('video-offline').style.display='block';">
        <div class="video-offline" id="video-offline" style="display:none;">CAMERA FEED UNAVAILABLE</div>
      </div>
      <div class="legend">
        <span><span class="swatch" style="background:#4ade80;"></span>Recognized</span>
        <span><span class="swatch" style="background:#f87171;"></span>Unknown / Alert</span>
        <span><span class="swatch" style="background:#fbbf24;"></span>Spoof suspected</span>
        <span><span class="swatch" style="background:#c8a03c;"></span>Zone boundary</span>
      </div>
    </div>

    <div class="side-col">
      <div class="events-panel">
        <div class="panel-title">Recent Events</div>
        <div class="events-scroll">
          <table id="events">
            <thead><tr><th>Time</th><th>Zone</th><th>Conf</th><th>Person</th><th>Status</th></tr></thead>
            <tbody></tbody>
          </table>
        </div>
      </div>

      <div class="enroll-panel">
        <div class="panel-title">Enroll Known Person</div>
        <div class="enroll-body">
          <p class="enroll-hint">Capture 3-5 photos from different angles for a robust match.</p>
          <div class="enroll-controls">
            <input type="text" id="enroll-name" placeholder="Person's name" maxlength="40">
            <button id="btn-capture" onclick="captureForEnroll()">Capture</button>
            <button id="btn-save" onclick="saveEnrollment()" disabled>Save (0)</button>
            <button id="btn-clear" onclick="clearEnrollment()">Clear</button>
          </div>
          <div class="thumbs" id="enroll-thumbs"></div>
          <div class="enroll-status" id="enroll-status"></div>
        </div>
      </div>
    </div>
  </main>

<script>
const evtSource = new EventSource("/stream");
evtSource.onmessage = function(e) {
  const data = JSON.parse(e.data);
  document.getElementById("stats").innerHTML = `
    <div class="s-item"><span class="s-label">Events</span><span class="s-value">${data.stats.total_events}</span></div>
    <div class="s-item"><span class="s-label">Alerts</span><span class="s-value">${data.stats.total_alerts}</span></div>
    <div class="s-item"><span class="s-label">24H</span><span class="s-value">${data.stats.events_24h}</span></div>
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
  btn.textContent = `Save (${stagedTokens.length})`;
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
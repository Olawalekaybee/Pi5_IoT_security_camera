/**
 * Pi-Detect — main.js
 * Polls /api/stats every second and updates live metrics panel.
 */

const POLL_MS = 1000;

const els = {
  statusDot:    document.getElementById("statusDot"),
  statusLabel:  document.getElementById("statusLabel"),
  statFps:      document.getElementById("statFps"),
  statInference: document.getElementById("statInference"),
  statCount:    document.getElementById("statCount"),
  objectList:   document.getElementById("objectList"),
  configGrid:   document.getElementById("configGrid"),
  videoFeed:    document.getElementById("videoFeed"),
  videoOverlay: document.getElementById("videoOverlay"),
};

let errors = 0;

async function fetchStats() {
  try {
    const res = await fetch("/api/stats", { cache: "no-store" });
    if (!res.ok) throw new Error();
    const data = await res.json();
    setOnline();
    errors = 0;
    updateMetrics(data);
  } catch {
    if (++errors >= 3) setOffline();
  }
}

function updateMetrics({ fps, inference_ms, detections, objects }) {
  els.statFps.textContent = fps ?? "—";
  els.statInference.innerHTML = inference_ms != null
    ? `${inference_ms}<span class="stat-unit">ms</span>` : "—";
  els.statCount.textContent = detections ?? "—";
  els.objectList.innerHTML = objects && objects.length
    ? objects.map(o => `<li class="object-item">${o}</li>`).join("")
    : `<li class="object-item object-item--empty">None detected</li>`;
}

function setOnline()  { els.statusDot.className = "status-dot online";  els.statusLabel.textContent = "Online"; }
function setOffline() { els.statusDot.className = "status-dot offline"; els.statusLabel.textContent = "Offline"; }

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    const data = await res.json();
    els.configGrid.innerHTML = Object.entries(data)
      .map(([k, v]) => `<div class="config-item"><div class="config-key">${k.replace(/_/g," ")}</div><div class="config-val">${v}</div></div>`)
      .join("");
  } catch { els.configGrid.innerHTML = `<p style="color:var(--text-muted)">Could not load config</p>`; }
}

function handleStreamError() {
  els.videoOverlay.classList.add("visible");
  setTimeout(() => {
    els.videoFeed.src = "/stream/video?" + Date.now();
    els.videoOverlay.classList.remove("visible");
  }, 3000);
}
window.handleStreamError = handleStreamError;

async function takeSnapshot() {
  const res  = await fetch("/stream/snapshot");
  const blob = await res.blob();
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = `snapshot_${Date.now()}.jpg`; a.click();
  URL.revokeObjectURL(url);
}
window.takeSnapshot = takeSnapshot;

function toggleFullscreen() {
  const w = document.querySelector(".video-wrapper");
  if (!document.fullscreenElement) w.requestFullscreen();
  else document.exitFullscreen();
}
window.toggleFullscreen = toggleFullscreen;

loadConfig();
fetchStats();
setInterval(fetchStats, POLL_MS);

/**
 * Pi-Detect — dashboard.js
 * Canvas-based FPS + detection history charts. No external chart library.
 */

const POLL_MS  = 1000;
const HISTORY  = 60;

const fpsSamples       = Array(HISTORY).fill(null);
const detectionSamples = Array(HISTORY).fill(null);
const freqMap          = {};

let totalDetections = 0;
let startTime       = Date.now();

const els = {
  statusDot:    document.getElementById("statusDot"),
  statusLabel:  document.getElementById("statusLabel"),
  kpiFps:       document.getElementById("kpiFps"),
  kpiInference: document.getElementById("kpiInference"),
  kpiTotal:     document.getElementById("kpiTotal"),
  kpiUptime:    document.getElementById("kpiUptime"),
  fpsCanvas:    document.getElementById("fpsChart"),
  detCanvas:    document.getElementById("detectionChart"),
  freqBody:     document.getElementById("freqTableBody"),
};

async function fetchStats() {
  try {
    const res  = await fetch("/api/stats", { cache: "no-store" });
    const data = await res.json();

    fpsSamples.push(data.fps);        fpsSamples.shift();
    detectionSamples.push(data.detections); detectionSamples.shift();
    totalDetections += data.detections;

    els.kpiFps.textContent       = data.fps ?? "—";
    els.kpiInference.innerHTML   = `${data.inference_ms ?? "—"}<small>ms</small>`;
    els.kpiTotal.textContent     = totalDetections;
    els.kpiUptime.textContent    = formatUptime(Date.now() - startTime);

    (data.objects || []).forEach(label => {
      if (!freqMap[label]) freqMap[label] = { count: 0, lastSeen: null };
      freqMap[label].count++;
      freqMap[label].lastSeen = new Date();
    });

    renderFpsChart();
    renderDetectionChart();
    renderFreqTable();

    els.statusDot.className = "status-dot online";
    els.statusLabel.textContent = "Online";
  } catch {
    els.statusDot.className = "status-dot offline";
    els.statusLabel.textContent = "Offline";
  }
}

function drawChart(canvas, samples, color, maxY) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W   = canvas.clientWidth;
  const H   = canvas.clientHeight;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  ctx.scale(dpr, dpr);

  const valid = samples.filter(v => v !== null);
  const max   = maxY ?? (valid.length ? Math.max(...valid) * 1.2 : 30);
  const step  = W / (HISTORY - 1);

  ctx.clearRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth   = 1;
  [0.25, 0.5, 0.75].forEach(f => {
    const y = H - f * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  });

  ctx.beginPath();
  samples.forEach((v, i) => {
    const x = i * step;
    const y = v !== null ? H - (v / max) * (H - 12) : H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, color + "55");
  grad.addColorStop(1, color + "00");
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.lineJoin    = "round";
  samples.forEach((v, i) => {
    const x = i * step;
    const y = v !== null ? H - (v / max) * (H - 12) : H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderFpsChart()       { drawChart(els.fpsCanvas, fpsSamples, "#00e5a0", 30); }
function renderDetectionChart() { drawChart(els.detCanvas, detectionSamples, "#f5a623"); }

function renderFreqTable() {
  const rows = Object.entries(freqMap).sort((a,b) => b[1].count - a[1].count).slice(0,15);
  if (!rows.length) return;
  els.freqBody.innerHTML = rows
    .map(([label, {count, lastSeen}]) =>
      `<tr><td>${label}</td><td>${count}</td><td>${lastSeen ? timeAgo(lastSeen) : "—"}</td></tr>`)
    .join("");
}

function formatUptime(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 60)   return `${s}s`;
  if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
}

function timeAgo(date) {
  const s = Math.floor((Date.now() - date) / 1000);
  if (s < 5)  return "just now";
  if (s < 60) return `${s}s ago`;
  return `${Math.floor(s/60)}m ago`;
}

fetchStats();
setInterval(fetchStats, POLL_MS);
window.addEventListener("resize", () => { renderFpsChart(); renderDetectionChart(); });

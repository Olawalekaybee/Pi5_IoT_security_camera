---
noteId: "9e3949306bf511f1a7af3f80fed50b71"
tags: []

---

# Pi5_IoT_security_camera

Real-time, fully offline person detection and re-identification system powered by the Hailo-8L NPU on a Raspberry Pi 5. No cloud, no subscription, no internet dependency — all inference runs at the edge.

[![CI](https://github.com/yourusername/edge-ai-security-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/edge-ai-security-monitor/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org)
[![Platform: Raspberry Pi 5](https://img.shields.io/badge/platform-Raspberry%20Pi%205-c51a4a.svg)](https://www.raspberrypi.com)

![Demo](docs/demo.gif)

---

## Overview

This project turns a Raspberry Pi 5 and a Hailo-8L AI HAT into a self-contained security appliance. A dual-model pipeline runs entirely on the NPU: YOLOv8 for person detection and a ResNet-based re-identification model for tracking individuals across frames and zones. When someone enters a restricted zone and isn't recognized as a known person, the system fires a Telegram alert with a snapshot, logs the event to a local database, and updates a live web dashboard — all without sending a single frame off the device.

It was built to demonstrate production-grade embedded AI engineering: real hardware, real constraints, real infrastructure — not a notebook demo.

## Why this exists

Most computer vision portfolio projects run on a laptop GPU or a cloud API and stop at "it detects objects." This project is different in three ways:

- **Real edge hardware.** Inference runs on a 26 TOPS NPU drawing under 5W, not a GPU or cloud endpoint.
- **Two models, one pipeline.** Detection and re-identification run concurrently on-device, with a custom cosine-similarity matching layer tying them together.
- **Shipped like production software.** Docker, systemd, CI, tests, and a documented model-conversion pipeline — not just a `main.py`.

## Features

| Capability | Detail |
|---|---|
| Person detection | YOLOv8 compiled to Hailo's HEF format, running on-NPU at ~30 FPS |
| Re-identification | ResNet embedding + cosine similarity to track identity across frames and camera zones |
| Zone awareness | Polygon-based restricted-zone definitions; alerts only fire inside configured zones |
| Alerting | Telegram bot integration with per-zone cooldown to prevent alert spam |
| Live dashboard | Flask + Server-Sent Events for real-time event streaming to a browser, no polling |
| Event logging | SQLite in WAL mode for concurrent read/write without blocking the inference loop |
| Hardware-free demo mode | CPU mock fallback so anyone can clone and run it without owning a Hailo HAT |
| Deployment | Docker Compose, systemd service for auto-start on boot, GitHub Actions CI |
| Tests | 22 unit tests, fully hardware-independent |

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Camera    │────▶│   Hailo-8L NPU   │────▶│   Zone & Re-ID   │
│   (CSI/USB) │     │  YOLOv8 + ResNet │     │      Logic       │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                        │
                          ┌─────────────────────────────┼─────────────────────────────┐
                          ▼                              ▼                              ▼
                  ┌───────────────┐            ┌──────────────────┐          ┌──────────────────┐
                  │  SQLite (WAL) │            │  Telegram Alert   │          │  Flask Dashboard │
                  │  Event Log    │            │  (per-zone cooldown)│        │  (SSE, live feed)│
                  └───────────────┘            └──────────────────┘          └──────────────────┘
```

**Inference flow:** camera frame → HailoRT VDevice → YOLOv8 person detection → crop detected persons → ResNet Re-ID embedding → cosine similarity match against known identities → zone-polygon check → event dispatch (log / alert / dashboard push).

## Hardware

| Component | Spec |
|---|---|
| Raspberry Pi 5 | 8GB RAM recommended |
| Hailo HAT+ | Hailo-8L, 26 TOPS |
| Camera | Pi Camera Module 3, or any USB/CSI camera |
| Storage | 32GB+ microSD (A2 rated recommended for SQLite WAL writes) |
| Power | Official Pi 5 27W USB-C supply |

No Hailo hardware? The CPU mock mode runs the full pipeline logic on ONNX Runtime so you can evaluate the system before buying anything.

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Olawalekaybee/Pi5_IoT_security_camera.git
cd edge-ai-security-monitor
pip install -r requirements.txt
```

### 2. Run in demo mode (no hardware required)

```bash
python run.py --mode mock --camera demo_footage/sample.mp4
```

### 3. Run on real hardware

```bash
python run.py --mode hailo --camera /dev/video0 --zones config/zones.yaml
```

### 4. Run with Docker

```bash
docker compose up -d
```

The dashboard will be available at `http://<device-ip>:5000`.

## Configuration

Zones, alert cooldowns, and model paths are defined in `config/`:

```yaml
# config/zones.yaml
zones:
  - name: "front_door"
    polygon: [[100, 200], [400, 200], [400, 480], [100, 480]]
    alert_cooldown_seconds: 60
```

Telegram credentials are set via environment variables (see `.env.example`):

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Model conversion (ONNX → HEF)

Hailo's Dataflow Compiler converts standard ONNX models to the Hailo Executable Format (HEF) required by the NPU. The conversion script and full walkthrough are in [`docs/model-conversion.md`](docs/model-conversion.md):

```bash
python scripts/convert_to_hef.py \
  --onnx models/yolov8n.onnx \
  --calib-dataset data/calibration/ \
  --output models/yolov8n.hef
```

## Benchmarks

Measured on Raspberry Pi 5 (8GB) + Hailo-8L HAT, 1080p input, person-detection workload:

| Metric | CPU only (mock mode) | Hailo-8L NPU |
|---|---|---|
| Inference latency | ~280 ms/frame | ~14 ms/frame |
| Throughput | ~3.5 FPS | ~30 FPS |
| Power draw under load | ~11W | ~8.5W |
| CPU utilization | ~95% | ~22% |

Full methodology in [`docs/benchmarks.md`](docs/benchmarks.md).

## Project structure

```
edge-ai-security-monitor/
├── src/
│   ├── inference/        # HailoRT pipeline, CPU mock fallback
│   ├── reid/              # Re-identification + cosine similarity matching
│   ├── zones/             # Polygon zone detection logic
│   ├── alerts/             # Telegram integration with cooldown
│   ├── dashboard/         # Flask app + SSE event streaming
│   └── storage/           # SQLite WAL event logging
├── scripts/
│   └── convert_to_hef.py  # ONNX → HEF model conversion
├── config/                # Zones, model paths, app settings
├── tests/                 # 12 hardware-free unit tests
├── docs/                  # Architecture, benchmarks, conversion guide
├── .github/workflows/     # CI pipeline
├── Dockerfile
├── docker-compose.yml
├── systemd/                # Auto-start service unit
└── requirements.txt
```

## Testing

```bash
pytest tests/ -v
```

All 22 tests run without Hailo hardware, using mocked inference outputs — CI runs the full suite on every push.

## Roadmap

- [ ] Multi-camera support with cross-camera Re-ID
- [ ] Web-based zone editor (draw polygons in-browser instead of YAML)
- [ ] Edge-to-edge mesh alerting across multiple Pi units
- [ ] Support for Hailo-8 (full, non-L variant) for higher-resolution streams

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built on [HailoRT](https://github.com/hailo-ai/hailort), [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), and the Hailo Model Zoo.

# Pi-Detect 🎯

> **Real-time AI object detection with live MJPEG streaming on a Raspberry Pi Zero 2W**  
> View annotated camera feeds from any browser, phone, or tablet — over WiFi or 4G — using YOLOv8n.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-purple)
![Tailscale](https://img.shields.io/badge/Remote-Tailscale-blue)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📸 Screenshots

| Live Stream with Detection | Analytics Dashboard |
|---------------------------|---------------------|
| Real-time annotated MJPEG stream with bounding boxes | FPS history, detection frequency charts |

> **Hardware used to build this project:**  
> Raspberry Pi Zero 2W + Pi Camera module + 4G LTE hotspot

---

## ✨ Features

- 🎥 **Live MJPEG stream** — viewable in any browser, VLC, or OpenCV client
- 🤖 **YOLOv8n detection** — 80 COCO object classes (person, car, phone, etc.)
- ⚡ **Async non-blocking detection** — stream runs at full FPS while detection runs in background thread
- 📊 **Real-time dashboard** — FPS history, inference time, object frequency charts
- 🌍 **Remote access** — Tailscale VPN for secure access from anywhere over 4G
- 🔄 **Auto-start on boot** — systemd service, zero terminal interaction needed
- 💻 **Dual mode** — PiCamera2 on hardware, OpenCV webcam for laptop development
- 📸 **Snapshot endpoint** — download single annotated JPEG frames

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Raspberry Pi Zero 2W                 │
│                                                      │
│  Pi Camera → PiCamera2 → CameraStream (thread)      │
│                              │                       │
│                         Frame buffer                 │
│                         ┌────┴────┐                  │
│                         │        │                   │
│                    Stream thread  Detector thread    │
│                    (30 FPS)      (YOLOv8n async)     │
│                         │        │                   │
│                         └────┬───┘                   │
│                              │                       │
│                         Flask Server :5000           │
│                         ├── /              (UI)      │
│                         ├── /dashboard    (charts)   │
│                         ├── /stream/video (MJPEG)    │
│                         ├── /stream/snapshot         │
│                         └── /api/stats   (JSON)      │
└─────────────────────────────────────────────────────┘
                              │
                         Tailscale VPN
                              │
              ┌───────────────┴───────────────┐
              │                               │
         Your Laptop                    Phone / Tablet
     http://100.x.x.x:5000         http://100.x.x.x:5000
```

---

## 📦 Project Structure

```
pi-detect/
├── main.py                        # Entry point
├── app/
│   ├── __init__.py                # App factory
│   ├── config.py                  # All configuration (env-driven)
│   ├── routes.py                  # Main routes + REST API
│   ├── detection/
│   │   └── detector.py            # Async YOLOv8n inference engine
│   ├── streaming/
│   │   ├── camera.py              # PiCamera2 / OpenCV abstraction
│   │   └── stream.py              # MJPEG stream blueprint
│   └── utils/
│       ├── logger.py              # Rotating file logger
│       └── metrics.py             # FPS / inference metrics
├── templates/
│   ├── index.html                 # Live stream viewer
│   └── dashboard.html             # Analytics dashboard
├── static/
│   ├── css/style.css              # Dark theme design system
│   └── js/
│       ├── main.js                # Stream page logic
│       └── dashboard.js           # Canvas charts
├── scripts/
│   ├── install_pi.sh              # One-shot Pi installer
│   ├── start.sh                   # Start server
│   └── pi-detect.service          # systemd unit file
├── tests/
│   └── test_app.py                # Smoke tests
├── models/                        # YOLOv8n weights (git-ignored)
├── logs/                          # Rotating logs (git-ignored)
├── requirements.txt               # Laptop deps
├── requirements-pi.txt            # Pi deps
└── .env.example                   # Config template
```

---

## 🔧 Hardware Requirements

| Component | Spec | Notes |
|-----------|------|-------|
| Board | Raspberry Pi Zero 2W | 1GHz quad-core, 512MB RAM |
| Camera | Pi Camera v1/v2/HQ | Connected via ribbon cable |
| Storage | microSD 16GB+ | Class 10 recommended |
| Power | 5V micro USB | Use PWR port, not OTG |
| Network | WiFi / 4G hotspot | For remote streaming |

---

## 🚀 Quick Start

### Option A — Laptop Development (no Pi needed)

```bash
# Clone
git clone https://github.com/Olawalekaybee/pi-detect.git
cd pi-detect

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install
pip install -r requirements-pi.txt

# Configure (use webcam)
cp .env.example .env
# Edit .env: set USE_PICAMERA=false

# Run
python main.py --debug
```

Open: **http://localhost:5000**

---

### Option B — Full Raspberry Pi Deployment

#### Step 1 — Flash the SD Card

1. Download **[Raspberry Pi Imager](https://www.raspberrypi.com/software/)**
2. Select device: **Raspberry Pi Zero 2W**
3. Select OS: **Raspberry Pi OS Lite (64-bit)**
4. Click the ⚙️ gear icon and configure:

```
Hostname:  pi-detect
Username:  pi
Password:  your-password
WiFi SSID: YourHotspotName     ← must match exactly (case-sensitive)
WiFi Pass: YourHotspotPassword
SSH:       Enable ✅
```

5. Flash and insert SD card into Pi

#### Step 2 — Boot and Connect

Power the Pi via the **PWR** micro USB port. Wait ~60 seconds for first boot.

Find the Pi's IP in your router/hotspot admin page, then SSH:

```bash
ssh pi@<pi-ip-address>
# Example: ssh pi@192.168.100.170
```

#### Step 3 — Clone and Install

```bash
# On the Pi
git clone https://github.com/Olawalekaybee/pi-detect.git
cd pi-detect

# Run one-shot installer (~15 mins)
bash scripts/install_pi.sh
```

#### Step 4 — Configure

```bash
nano .env
```

Key settings:
```env
USE_PICAMERA=true
CAMERA_WIDTH=320
CAMERA_HEIGHT=240
DETECTION_SKIP_FRAMES=10
CONFIDENCE_THRESHOLD=0.35
```

#### Step 5 — Run

```bash
python3 main.py
```

Open on your laptop: **http://\<pi-ip\>:5000**

---

## 🌍 Remote Access via Tailscale

Access your stream from **anywhere in the world** over 4G.

### Setup (one-time)

**On the Pi:**
```bash
# Fix /tmp permissions first
sudo chmod 1777 /tmp

# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# Open the printed URL in your browser and sign in
```

**Get your Pi's permanent IP:**
```bash
tailscale ip
# Example output: 100.72.1XX.XXX
```

**On your laptop/phone:**
- Download [Tailscale](https://tailscale.com/download) and sign in with the same account

**Access from anywhere:**
```
http://100.72.1XX.XXX:5000
```

This IP **never changes** — bookmark it!

---

## 🔄 Auto-Start on Boot

```bash
# Copy service file
sudo cp scripts/pi-detect.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable pi-detect
sudo systemctl start pi-detect

# Check status
sudo systemctl status pi-detect
```

After setup, the stream starts **automatically** every time the Pi powers on — no terminal needed.

**Useful commands:**
```bash
sudo systemctl status pi-detect      # Check status
sudo systemctl restart pi-detect     # Restart
sudo journalctl -u pi-detect -f      # Live logs
tail -f logs/service.log             # App logs
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Live stream viewer UI |
| `/dashboard` | GET | Analytics dashboard |
| `/stream/video` | GET | MJPEG stream |
| `/stream/snapshot` | GET | Single JPEG frame |
| `/api/stats` | GET | Live metrics JSON |
| `/api/config` | GET | Current config JSON |
| `/api/health` | GET | Health check |

**Stats response example:**
```json
{
  "fps": 29.8,
  "detections": 1,
  "inference_ms": 1191.6,
  "detection_enabled": true,
  "objects": ["person"]
}
```

---

## ⚙️ Configuration Reference

All settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_PICAMERA` | `true` | `true` = PiCamera2, `false` = webcam |
| `CAMERA_WIDTH` | `320` | Frame width in pixels |
| `CAMERA_HEIGHT` | `240` | Frame height in pixels |
| `CAMERA_FPS` | `30` | Target capture FPS |
| `MODEL_PATH` | `models/yolov8n.pt` | Path to YOLOv8 weights |
| `CONFIDENCE_THRESHOLD` | `0.35` | Min detection confidence |
| `DETECTION_SKIP_FRAMES` | `10` | Run detection every N frames |
| `STREAM_QUALITY` | `50` | JPEG quality (1-100) |
| `STREAM_MAX_FPS` | `12` | Max stream FPS |
| `DEVICE` | `cpu` | Inference device |

### Performance Tuning (Pi Zero 2W)

| Setting | Conservative | Balanced | Performance |
|---------|-------------|----------|-------------|
| `DETECTION_SKIP_FRAMES` | `15` | `10` | `5` |
| `STREAM_QUALITY` | `40` | `50` | `70` |
| `STREAM_MAX_FPS` | `8` | `12` | `15` |
| `CAMERA_WIDTH/HEIGHT` | `320x240` | `320x240` | `640x480` |
| `CONFIDENCE_THRESHOLD` | `0.5` | `0.35` | `0.25` |

---

## 🖥️ View Stream in Other Clients

**VLC:**
```
Media → Open Network Stream → http://<ip>:5000/stream/video
```

**OpenCV Python:**
```python
import cv2
cap = cv2.VideoCapture("http://100.72.141.123:5000/stream/video")
while True:
    ret, frame = cap.read()
    cv2.imshow("Pi-Detect", frame)
    if cv2.waitKey(1) == ord("q"):
        break
```

**Browser (any device):**
```
http://100.72.XXX.XXX:5000
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Pi not showing on hotspot | Check WiFi name is case-exact in .env |
| `/tmp` permission errors | Run `sudo chmod 1777 /tmp` |
| PyTorch install fails | Run `sudo mount --bind ~/tmp /tmp` first |
| Stream shows static image | Service running as wrong user — set `User=pi` in service file |
| Detection says "dog" not "person" | Improve lighting — YOLOv8 struggles in dark/blue-tinted light |
| SSH timeout | Don't switch hotspots during install |

---

## 📊 Performance Benchmarks (Pi Zero 2W)

| Metric | Value |
|--------|-------|
| Stream FPS | ~30 FPS |
| Detection inference | ~1200ms per frame |
| Effective detection rate | ~1 detection/sec |
| RAM usage | ~380MB / 512MB |
| CPU usage | ~85-95% |

> **Tip:** The async detection thread means stream FPS is always smooth regardless of inference time.

---

## 🗺️ Roadmap

- [ ] Motion detection alerts (WhatsApp / email)
- [ ] Mobile-optimised UI
- [ ] Custom model training support
- [ ] Object tracking with IDs
- [ ] Recording / timelapse mode
- [ ] Multi-camera support

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Olawale** — Built end-to-end on a Raspberry Pi Zero 2W  
GitHub: [@Olawalekaybee](https://github.com/Olawalekaybee)

---

<div align="center">
  <strong>⭐ Star this repo if it helped you!</strong>
</div>

"""
MJPEG streaming blueprint.
/stream/video  — multipart MJPEG stream (browser, VLC, OpenCV compatible)
/stream/snapshot — single JPEG frame
"""

import time
import cv2
from flask import Blueprint, Response, current_app
from app.config import Config

stream_bp = Blueprint("stream", __name__, url_prefix="/stream")
_BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
_MIN_INTERVAL = 1.0 / Config.STREAM_MAX_FPS


def _draw_boxes(frame, boxes: list):
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        label = box["label"]
        conf = box["confidence"]
        color = box.get("color", (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {len(boxes)}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def _generate_frames(app):
    camera = app.camera
    detector = app.detector
    detection_enabled = app.config["DETECTION_ENABLED"]
    skip = Config.DETECTION_SKIP_FRAMES
    frame_index = 0
    last_sent = 0.0

    while True:
        now = time.time()
        if now - last_sent < _MIN_INTERVAL:
            time.sleep(0.005)
            continue

        frame = camera.read()
        if frame is None:
            time.sleep(0.02)
            continue

        if detection_enabled and detector is not None:
            if frame_index % skip == 0:
                detector.submit_frame(frame)
            boxes = detector.get_boxes()
            _draw_boxes(frame, boxes)

        frame_index += 1
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, Config.STREAM_QUALITY]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            continue

        last_sent = time.time()
        yield _BOUNDARY + buffer.tobytes() + b"\r\n"


@stream_bp.route("/video")
def video_feed():
    app = current_app._get_current_object()
    return Response(
        _generate_frames(app),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@stream_bp.route("/snapshot")
def snapshot():
    app = current_app._get_current_object()
    frame = app.camera.read()
    if frame is None:
        return Response("Camera not ready", status=503)
    if app.detector and app.config["DETECTION_ENABLED"]:
        boxes = app.detector.get_boxes()
        _draw_boxes(frame, boxes)
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buffer.tobytes(), mimetype="image/jpeg")

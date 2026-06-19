#!/usr/bin/env python3
"""
Converts an ONNX model (e.g. exported YOLOv8) into Hailo's .hef format
using the Hailo Dataflow Compiler (DFC). Must be run on a machine with
the Hailo SDK installed (typically a dev machine, not the Pi itself).

Usage:
    python scripts/convert_to_hef.py \\
        --onnx models/yolov8n.onnx \\
        --output models/yolov8n.hef \\
        --calib-data data/calibration_images/
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Convert ONNX model to Hailo HEF format")
    p.add_argument("--onnx", required=True, help="Path to input .onnx model")
    p.add_argument("--output", required=True, help="Path to output .hef file")
    p.add_argument("--calib-data", required=True,
                    help="Directory of representative images for INT8 calibration")
    p.add_argument("--hw-arch", default="hailo8l",
                    choices=["hailo8", "hailo8l", "hailo15"],
                    help="Target Hailo chip architecture")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from hailo_sdk_client import ClientRunner
    except ImportError:
        logger.error(
            "hailo_sdk_client not found. Install the Hailo Dataflow Compiler "
            "(available from the Hailo Developer Zone) on this machine — "
            "it is NOT the same package as hailo_platform used on the Pi."
        )
        sys.exit(1)

    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    calib_dir = Path(args.calib_data)

    if not onnx_path.exists():
        logger.error(f"ONNX model not found: {onnx_path}")
        sys.exit(1)
    if not calib_dir.exists() or not any(calib_dir.iterdir()):
        logger.error(f"Calibration directory empty or missing: {calib_dir}")
        sys.exit(1)

    logger.info(f"Translating {onnx_path.name} for {args.hw_arch}...")
    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(str(onnx_path), onnx_path.stem)

    logger.info("Running post-translation optimization (INT8 quantization)...")
    calib_images = _load_calibration_set(calib_dir)
    runner.optimize(calib_images)

    logger.info("Compiling to HEF...")
    hef = runner.compile()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(hef)

    logger.info(f"Done. HEF written to {output_path}")
    logger.info(
        "Copy this .hef file to the Raspberry Pi's models/ directory "
        "and reference it in config/settings.yaml."
    )


def _load_calibration_set(calib_dir: Path, max_images: int = 64):
    import cv2
    import numpy as np

    images = []
    paths = sorted(calib_dir.glob("*.jpg")) + sorted(calib_dir.glob("*.png"))
    for p in paths[:max_images]:
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.resize(img, (640, 640))
            images.append(img.astype(np.float32) / 255.0)

    if not images:
        raise RuntimeError(f"No valid calibration images found in {calib_dir}")

    logger.info(f"Loaded {len(images)} calibration images")
    return images


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/convert_models.py
Convert YOLOv8 + Re-ID models from ONNX to Hailo HEF format.

Requires: Hailo Dataflow Compiler (hailomz) — available in Hailo Developer Zone.
Steps documented here match Hailo SDK 4.x.

Usage (run on your development machine, NOT the Pi):
  python scripts/convert_models.py --model yolov8n --output models/
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


def convert_yolov8(output_dir: Path, model_size: str = "n"):
    """
    YOLOv8n → ONNX → Hailo HEF (Hailo-8L target)

    Step 1: Export from Ultralytics to ONNX
    Step 2: Parse ONNX with Hailo Model Zoo compiler
    Step 3: Optimize + quantize for Hailo-8L
    Step 4: Compile to .hef
    """
    onnx_path = output_dir / f"yolov8{model_size}.onnx"
    hef_path = output_dir / f"yolov8{model_size}.hef"

    # 1. Export ONNX (needs ultralytics installed)
    run(
        [
            "python3", "-c",
            f"from ultralytics import YOLO; "
            f"m = YOLO('yolov8{model_size}.pt'); "
            f"m.export(format='onnx', opset=11, dynamic=False, imgsz=640)",
        ],
        "Exporting YOLOv8 to ONNX",
    )

    # 2-4. Hailo Model Zoo compile pipeline
    # hailomz is the Hailo Model Zoo CLI tool
    run(
        [
            "hailomz", "compile",
            "--hw-arch", "hailo8l",         # Target: Hailo-8L (HAT+ chip)
            "--onnx", str(onnx_path),
            "--output-dir", str(output_dir),
            "--calib-path", "data/calibration",  # ~200 sample images for quantisation
            "--performance",                # Optimise for throughput
        ],
        "Compiling YOLOv8 to HEF for Hailo-8L",
    )

    if hef_path.exists():
        size_mb = hef_path.stat().st_size / 1e6
        print(f"\n✓ HEF written: {hef_path} ({size_mb:.1f} MB)")
    else:
        print(f"\n✗ HEF not found at expected path: {hef_path}")
        sys.exit(1)


def convert_reid(output_dir: Path):
    """
    ResNet50 Re-ID → ONNX → HEF
    Uses a pretrained market1501 checkpoint.
    """
    hef_path = output_dir / "resnet50_reid.hef"

    run(
        [
            "hailomz", "compile",
            "--hw-arch", "hailo8l",
            "--model-name", "resnet_v1_50",
            "--output-dir", str(output_dir),
        ],
        "Compiling ResNet50 Re-ID to HEF",
    )

    if hef_path.exists():
        print(f"\n✓ Re-ID HEF written: {hef_path}")
    else:
        print(f"\n  Note: rename output file to resnet50_reid.hef")


def main():
    parser = argparse.ArgumentParser(description="Convert models to Hailo HEF format")
    parser.add_argument("--model", default="n",
                        choices=["n", "s", "m"],
                        help="YOLOv8 model size (n=nano, s=small, m=medium)")
    parser.add_argument("--output", default="models/",
                        help="Output directory for .hef files")
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-reid", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_yolo:
        convert_yolov8(output_dir, args.model)

    if not args.skip_reid:
        convert_reid(output_dir)

    print("\n✓ All models compiled. Copy .hef files to your Pi 5.")


if __name__ == "__main__":
    main()

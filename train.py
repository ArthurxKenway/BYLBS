"""
Train a custom YOLOv8 model on your Roboflow stationery dataset.

Usage:
  1. Download a YOLOv8-format dataset from Roboflow Universe
  2. Unzip it and note the path to data.yaml
  3. Run: python train.py --data path/to/data.yaml

Or copy this script into Google Colab and set DATA_YAML there.
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).parent
DEFAULT_OUTPUT = ROOT_DIR / "best.pt"


def default_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def train(data_yaml: str, epochs: int, imgsz: int, device: str) -> Path:
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
    )

    weights_dir = Path(results.save_dir) / "weights"
    best_weights = weights_dir / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Training finished but {best_weights} was not found.")

    shutil.copy2(best_weights, DEFAULT_OUTPUT)
    print(f"\nCopied trained weights to {DEFAULT_OUTPUT}")
    print(f"Model classes: {model.names}")
    return DEFAULT_OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a stationery dataset")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the Roboflow data.yaml file",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default=default_device(),
        help="GPU index (0) or 'cpu' (auto-detects if omitted)",
    )
    args = parser.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        print("No CUDA GPU detected — training on CPU (this will be slower).")
        print("For faster training, use Google Colab with a free GPU.\n")
        args.device = "cpu"

    train(args.data, args.epochs, args.imgsz, args.device)


if __name__ == "__main__":
    main()

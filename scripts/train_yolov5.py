"""Train YOLOv5 for microparticle detection using the thesis configuration."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Train YOLOv5 on the microparticle dataset")
    parser.add_argument("--data", default="data/data.yaml", help="Path to the YOLOv5 data config")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per device")
    parser.add_argument(
        "--weights",
        default=os.environ.get("YOLOV5_WEIGHTS", "yolov5s.pt"),
        help="Initial weights checkpoint (default env YOLOV5_WEIGHTS or yolov5s.pt)",
    )
    parser.add_argument(
        "--hyp",
        default="yolov5_hyp_no_rotate.yaml",
        help="Hyperparameter YAML (default: %(default)s)",
    )
    parser.add_argument("--project", default="runs/train", help="YOLOv5 project directory")
    parser.add_argument("--name", default="mp_yolov5", help="YOLOv5 experiment name")
    parser.add_argument(
        "--device",
        default="",
        help="Torch device string ('' for auto, e.g. 'cpu', '0', '0,1')",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to yolov5/train.py (prefix with --)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    """Configure the Python logging module."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_training(args: argparse.Namespace) -> None:
    """Execute YOLOv5's training script with the provided arguments."""

    setup_logging(args.log_level)
    yolov5_dir = PROJECT_ROOT / "yolov5"
    train_script = yolov5_dir / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            "YOLOv5 train.py not found. Ensure the submodule is initialised via scripts/setup_yolov5.sh",
        )

    command = [
        sys.executable,
        str(train_script),
        "--img",
        str(args.imgsz),
        "--batch",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--data",
        str(args.data),
        "--weights",
        str(args.weights),
        "--project",
        str(args.project),
        "--name",
        str(args.name),
        "--hyp",
        str(args.hyp),
    ]

    if args.device:
        command.extend(["--device", str(args.device)])
    if args.extra_args:
        command.extend(args.extra_args)

    logging.info("Executing: %s", " ".join(command))
    subprocess.run(command, check=True)


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


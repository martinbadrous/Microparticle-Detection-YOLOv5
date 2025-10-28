"""Split raw microscopy images into train/val/test YOLOv5 folders."""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path
import sys
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import IMAGE_EXTENSIONS, list_images  # noqa: E402


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Split a dataset into train/val/test sets")
    parser.add_argument("--images-dir", default="data/images_all", help="Directory containing all images")
    parser.add_argument(
        "--labels-dir",
        default="data/labels_all",
        help="Directory containing YOLO label files (optional)",
    )
    parser.add_argument("--out-root", default="data", help="Output root containing images/ and labels/")
    parser.add_argument("--train", type=float, default=0.7, help="Proportion of images for the train split")
    parser.add_argument("--val", type=float, default=0.2, help="Proportion of images for the val split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    """Configure logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def copy_image_and_label(image_path: Path, dst_images: Path, labels_dir: Path, dst_labels: Path) -> None:
    """Copy a single image and its label (if it exists) to the destination folders."""

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, dst_images / image_path.name)
    label_path = labels_dir / f"{image_path.stem}.txt"
    if label_path.exists():
        shutil.copy2(label_path, dst_labels / label_path.name)


def perform_split(args: argparse.Namespace) -> None:
    """Execute the dataset split."""

    setup_logging(args.log_level)
    random.seed(args.seed)

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_root = Path(args.out_root)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    images = list_images(images_dir, extensions=IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    indices = list(range(len(images)))
    random.shuffle(indices)

    n_train = int(len(images) * args.train)
    n_val = int(len(images) * args.val)
    train_indices = set(indices[:n_train])
    val_indices = set(indices[n_train : n_train + n_val])
    test_indices = set(indices[n_train + n_val :])

    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    logging.info(
        "Splitting %s images into train=%s, val=%s, test=%s",
        len(images),
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    image_paths = [Path(p) for p in images]
    for split_name, idx_set in splits.items():
        dst_images = out_root / "images" / split_name
        dst_labels = out_root / "labels" / split_name
        for idx, image_path in enumerate(image_paths):
            if idx in idx_set:
                copy_image_and_label(image_path, dst_images, labels_dir, dst_labels)

    logging.info("Dataset split complete. Files copied to %s", out_root)


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    perform_split(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


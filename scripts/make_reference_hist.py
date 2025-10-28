"""Generate the reference histogram used for microparticle classification."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Iterable, Optional

import cv2 as cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.postproc import enhance_contrast_pointwise, hist_256  # noqa: E402

LOGGER = logging.getLogger("make_reference_hist")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create the reference histogram from a well-focused particle crop",
    )
    parser.add_argument("--crop", required=True, help="Path to an optimal particle crop (grayscale or BGR)")
    parser.add_argument(
        "--out",
        default="reference_hist.json",
        help="Output JSON path (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.2,
        help="Contrast alpha for point-wise enhancement (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    """Configure logging for the script."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    setup_logging(args.log_level)

    crop_path = Path(args.crop)
    if not crop_path.exists():
        raise FileNotFoundError(f"Crop image not found: {crop_path}")

    image = cv.imread(str(crop_path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read crop image: {crop_path}")

    LOGGER.info("Loaded crop %s with shape %s", crop_path, image.shape)
    image_enhanced = enhance_contrast_pointwise(image, alpha=args.alpha)
    histogram = hist_256(image_enhanced, normalize=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({"alpha": args.alpha, "hist": histogram.tolist()}, handle, indent=2)
    LOGGER.info("Saved reference histogram to %s", out_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


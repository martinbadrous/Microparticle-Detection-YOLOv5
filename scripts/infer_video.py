"""Run YOLOv5 inference and extract microparticle features from a video."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Iterable, Optional

import cv2 as cv
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.postproc import (  # noqa: E402
    chi_square_distance,
    enhance_contrast_pointwise,
    hist_256,
    laplacian_variance,
    otsu_area_px,
)
from src.yolo import YOLOv5Detector  # noqa: E402

LOGGER = logging.getLogger("infer_video")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="YOLOv5-based microparticle detection with thesis post-processing",
    )
    parser.add_argument("--weights", required=True, help="Path to YOLOv5 .pt weights")
    parser.add_argument("--video", required=True, help="Path to the input microscopy video")
    parser.add_argument(
        "--ref-hist-json",
        required=True,
        help="Reference histogram JSON produced by make_reference_hist.py",
    )
    parser.add_argument(
        "--out-dir",
        default="reports",
        help="Directory where the CSV report will be written (default: %(default)s)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional explicit CSV output path (default: <out-dir>/features_3d.csv)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold used during non-maximum suppression",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=None,
        help="Restrict inference to specific class indices (e.g. 0 1)",
    )
    parser.add_argument(
        "--min-area-px",
        type=int,
        default=500,
        help="Minimum Otsu foreground area (px) to keep a detection",
    )
    parser.add_argument(
        "--um2-per-px",
        type=float,
        default=0.30,
        help="Conversion factor from pixels to µm² (set 0 to disable conversion)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override the histogram contrast alpha (defaults to reference JSON)",
    )
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections per frame")
    parser.add_argument(
        "--device",
        default="",
        help="Torch device string ('' for auto, e.g. 'cpu', '0', '0,1')",
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


def load_reference_histogram(path: Path, alpha_override: Optional[float]) -> tuple[np.ndarray, float]:
    """Load the thesis reference histogram JSON file."""

    if not path.exists():
        raise FileNotFoundError(f"Reference histogram file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        ref = json.load(handle)
    alpha = alpha_override if alpha_override is not None else float(ref.get("alpha", 1.2))
    hist = np.asarray(ref["hist"], dtype=np.float64)
    hist /= float(hist.sum() + 1e-8)
    return hist, alpha


def compute_features(
    frame_index: int,
    bbox: tuple[int, int, int, int],
    class_name: str,
    confidence: float,
    crop_bgr: np.ndarray,
    min_area_px: int,
    ref_hist: np.ndarray,
    ref_alpha: float,
    um2_per_px: float,
) -> Optional[dict[str, object]]:
    """Compute thesis features for a YOLO detection crop."""

    if crop_bgr.size == 0:
        LOGGER.debug("Skipping empty crop at frame %s", frame_index)
        return None

    gray = cv.cvtColor(crop_bgr, cv.COLOR_BGR2GRAY)
    area_px, _ = otsu_area_px(gray)
    if area_px < min_area_px:
        LOGGER.debug(
            "Frame %s detection filtered: area_px=%s < min_area_px=%s",
            frame_index,
            area_px,
            min_area_px,
        )
        return None

    blur_var = laplacian_variance(gray)
    gray_ce = enhance_contrast_pointwise(gray, alpha=ref_alpha)
    hist = hist_256(gray_ce, normalize=True)
    chi2 = chi_square_distance(hist, ref_hist)
    area_um2 = area_px * um2_per_px if um2_per_px > 0 else None

    x1, y1, x2, y2 = bbox
    return {
        "frame": frame_index,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": x2 - x1,
        "h": y2 - y1,
        "cls": class_name,
        "conf": confidence,
        "area_px": area_px,
        "area_um2": area_um2,
        "blur_var": blur_var,
        "chi2": chi2,
    }


def infer_video(args: argparse.Namespace) -> Path:
    """Run the full detection and feature extraction pipeline."""

    setup_logging(args.log_level)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv) if args.output_csv else out_dir / "features_3d.csv"

    ref_hist, ref_alpha = load_reference_histogram(Path(args.ref_hist_json), args.alpha)
    LOGGER.info("Loaded reference histogram from %s (alpha=%.3f)", args.ref_hist_json, ref_alpha)

    try:
        detector = YOLOv5Detector(
            weights=Path(args.weights),
            imgsz=args.imgsz,
            conf_thres=args.conf,
            iou_thres=args.iou,
            classes=args.classes,
            device=args.device,
            max_det=args.max_det,
        )
    except ModuleNotFoundError:  # pragma: no cover - dependency error
        LOGGER.error("YOLOv5 is not available. Run scripts/setup_yolov5.sh first.")
        raise
    LOGGER.info(
        "Detector ready (weights=%s, imgsz=%s, conf=%.2f, iou=%.2f)",
        args.weights,
        args.imgsz,
        args.conf,
        args.iou,
    )

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    records: list[dict[str, object]] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_idx += 1
            detections = detector.predict(frame)
            if frame_idx % 25 == 0:
                LOGGER.debug("Processing frame %s (detections=%s)", frame_idx, len(detections))

            height, width = frame.shape[:2]
            for det in detections:
                x1 = max(0, min(det.xmin, width))
                y1 = max(0, min(det.ymin, height))
                x2 = max(0, min(det.xmax, width))
                y2 = max(0, min(det.ymax, height))
                if x2 <= x1 or y2 <= y1:
                    LOGGER.debug(
                        "Invalid bbox after clipping (frame=%s, bbox=%s)",
                        frame_idx,
                        (det.xmin, det.ymin, det.xmax, det.ymax),
                    )
                    continue
                crop = frame[y1:y2, x1:x2]
                features = compute_features(
                    frame_index=frame_idx,
                    bbox=(x1, y1, x2, y2),
                    class_name=det.class_name,
                    confidence=det.confidence,
                    crop_bgr=crop,
                    min_area_px=args.min_area_px,
                    ref_hist=ref_hist,
                    ref_alpha=ref_alpha,
                    um2_per_px=args.um2_per_px,
                )
                if features:
                    records.append(features)
    finally:
        capture.release()

    if not records:
        LOGGER.warning("No valid detections were recorded. CSV will be empty.")

    df = pd.DataFrame.from_records(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    LOGGER.info("Wrote %s (%d rows)", output_csv, len(df))
    return output_csv


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    infer_video(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


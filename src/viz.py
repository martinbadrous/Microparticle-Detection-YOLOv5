"""Visualisation helpers for debugging detections."""

from __future__ import annotations

from typing import Optional

import cv2 as cv


def draw_box(
    img,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: Optional[str] = None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> "cv.Mat":
    """Draw a labelled bounding box on ``img`` in-place."""

    cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv.LINE_AA,
        )
    return img


__all__ = ["draw_box"]


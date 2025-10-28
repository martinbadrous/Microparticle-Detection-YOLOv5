"""Post-processing utilities used by the microparticle thesis pipeline."""

from __future__ import annotations

from typing import Tuple

import cv2 as cv
import numpy as np


def otsu_area_px(img_gray: np.ndarray) -> Tuple[int, np.ndarray]:
    """Compute foreground area (in pixels) using Otsu's threshold.

    Parameters
    ----------
    img_gray:
        Single-channel 8-bit grayscale image.

    Returns
    -------
    area_px, mask:
        ``area_px`` is the number of foreground pixels, ``mask`` is the binary
        mask produced by Otsu's method (with automatic inversion when the
        foreground is darker than the background).
    """

    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
    _, mask = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    if float(np.mean(mask)) < 127:
        mask = cv.bitwise_not(mask)
    area = int(np.count_nonzero(mask == 255))
    return area, mask


def laplacian_variance(img_gray: np.ndarray) -> float:
    """Return the Laplacian variance focus metric for ``img_gray``."""

    lap = cv.Laplacian(img_gray, cv.CV_64F, ksize=3)
    return float(lap.var())


def hist_256(img_gray: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute a 256-bin grayscale histogram.

    The histogram is optionally L1-normalised to sum to one, which is required
    for the chi-square distance used in the thesis.
    """

    hist = cv.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    if normalize:
        total = float(hist.sum()) + 1e-8
        hist = hist / total
    return hist


def chi_square_distance(h1: np.ndarray, h2: np.ndarray, eps: float = 1e-8) -> float:
    """Return the chi-square distance between histograms ``h1`` and ``h2``."""

    num = (h1 - h2) ** 2
    den = h1 + h2 + eps
    return float(np.sum(num / den))


def enhance_contrast_pointwise(
    img_gray: np.ndarray, alpha: float = 1.2, beta: float = 0
) -> np.ndarray:
    """Apply simple point-wise contrast enhancement used in the thesis."""

    return cv.convertScaleAbs(img_gray, alpha=alpha, beta=beta)


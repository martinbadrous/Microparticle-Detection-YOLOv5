"""Utility helpers for loading and running YOLOv5 models.

The module keeps the YOLOv5 dependency local to this repository by adding
``<repo_root>/yolov5`` to ``sys.path``.  The logic mirrors the inference steps
from ``detect.py`` in YOLOv5 v7 while exposing a small, well documented API
that can be reused by the thesis scripts.

The loader intentionally raises clear exceptions when the YOLOv5 submodule is
missing so that users immediately know they have to run ``scripts/setup_yolov5``
before training or inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
YOLOV5_DIR = REPO_ROOT / "yolov5"
if YOLOV5_DIR.exists() and str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

try:  # Lazily import YOLOv5 internals once the path is configured.
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.datasets import letterbox
    from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
    from yolov5.utils.torch_utils import select_device
except ModuleNotFoundError as exc:  # pragma: no cover - raised in __getattr__
    raise ModuleNotFoundError(
        "YOLOv5 modules not found. Did you run scripts/setup_yolov5.sh?"
    ) from exc


@dataclass
class Detection:
    """Container for a single YOLOv5 detection result."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float
    class_id: int
    class_name: str


class YOLOv5Detector:
    """Thin wrapper around :class:`DetectMultiBackend` for frame-level inference."""

    def __init__(
        self,
        weights: Path,
        imgsz: int = 640,
        conf_thres: float = 0.5,
        iou_thres: float = 0.5,
        classes: Optional[Iterable[int]] = None,
        device: str = "",
        max_det: int = 1000,
        half: bool = False,
    ) -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights}")

        self.repo_root = REPO_ROOT
        self.yolov5_dir = YOLOV5_DIR
        if not self.yolov5_dir.exists():
            raise FileNotFoundError(
                "YOLOv5 submodule is missing. Run 'scripts/setup_yolov5.sh' first."
            )

        self.device = select_device(device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False)
        self.stride = int(self.model.stride)
        self.imgsz = check_img_size((imgsz, imgsz), s=self.stride)
        self.names = self.model.names
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = list(classes) if classes is not None else None
        self.max_det = max_det
        self.half = half and self.model.fp16

        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame.

        Parameters
        ----------
        frame:
            A ``numpy.ndarray`` in BGR colour order as returned by OpenCV.

        Returns
        -------
        list of :class:`Detection`
            Bounding boxes scaled back to the input frame resolution.
        """

        if frame is None or frame.size == 0:
            return []

        img = letterbox(frame, self.imgsz, stride=self.stride, auto=self.model.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to CHW
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        pred = self.model(tensor)
        pred = non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            max_det=self.max_det,
        )

        detections: List[Detection] = []
        for det in pred:
            if det is None or not len(det):
                continue
            det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det.tolist():
                xmin, ymin, xmax, ymax = map(int, xyxy)
                class_id = int(cls)
                detections.append(
                    Detection(
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                        confidence=float(conf),
                        class_id=class_id,
                        class_name=str(self.names[class_id]),
                    )
                )
        return detections


__all__ = ["Detection", "YOLOv5Detector"]


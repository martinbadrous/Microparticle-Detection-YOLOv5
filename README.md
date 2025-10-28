# Microparticle-Detection-YOLOv5

End-to-end YOLOv5 pipeline to detect and count two spherical microplastics classes: `PMMA10` and `PMMA20` in microscope videos. 
It reproduces the thesis approach: YOLOv5 detection → per-particle features (Otsu area, Laplacian variance blur, Chi-square histogram distance) → 3D scatter analysis → counts.

## Quick start

```bash
# 1) Clone and set up
git clone https://github.com/<you>/Microparticle-Detection-YOLOv5.git
cd Microparticle-Detection-YOLOv5
bash scripts/setup_yolov5.sh

# 2) (Optional) create/activate venv and install
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) Put your images and labels (YOLO format) under data/
# If you don't have labels, annotate with a tool (LabelImg/Roboflow) using class names:
# PMMA10, PMMA20

# Expected structure (YOLO):
# data/
#   images/{train,val,test}/*.jpg|*.png
#   labels/{train,val,test}/*.txt

# 4) Update dataset config if needed
# (defaults already set to classes PMMA10, PMMA20)
# data/data.yaml

# 5) Train
python scripts/train_yolov5.py

# 6) Build a reference histogram from your best in-focus particle crop
python scripts/make_reference_hist.py --crop path/to/best_crop.png --out reference_hist.json

# 7) Run inference on a video and compute features
python scripts/infer_video.py --weights runs/train/mp_yolov5/weights/best.pt   --video path/to/video.mp4   --ref_hist_json reference_hist.json   --um2_per_px 0.30

# 8) Plot 3D scatter of (Area_px, BlurVar, Chi2)
python scripts/plot_3d_features.py --csv reports/features_3d.csv

# 9) Evaluate (optional; YOLOv5 produces PR/Recall/F1/mAP and confusion matrix under runs/)
```

## Dataset config (`data/data.yaml`)

- Classes: `PMMA10`, `PMMA20`
- Split: recommended 70/20/10 (train/val/test)
- Image/label folders must mirror each other

## Post-processing features

For each detection:
- **Area (px)** via Otsu threshold of detection crop (white pixel count)
- **Blur** via Laplacian variance
- **Histogram distance**: Chi-square against a reference histogram (from a chosen “optimal” particle crop)

Detections with **area < 500 px** are filtered out.

## Units

- Pixel area to µm²: **0.30 µm²/pixel**

## Notes

- Default model: **YOLOv5s**. You can switch to `yolov5m.pt` in `scripts/train_yolov5.py` if you expand the dataset.
- Augmentations: strong photometric + blur/noise; rotations disabled (spherical particles).
- Video defaults: microscope videos at ~720p/30 fps/20× objective work well.

## Citing

Please cite the associated thesis when using this code.

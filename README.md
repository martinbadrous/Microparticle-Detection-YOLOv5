
# 🧫 High-Speed Microparticle Detection and Tracking using YOLOv5

This repository implements the **deep learning-based microparticle detection pipeline** from my Master's thesis, titled  
**“High-Speed Microparticle Detection and Tracking” (2021)**,  
completed under the **ViBOT Master’s Program, Bourgogne University**.

It extends my previous classical computer vision work ([Microparticle-Detection](https://github.com/martinbadrous/Microparticle-Detection)) by introducing a **YOLOv5-based deep learning approach** for accurate detection and counting of PMMA microplastic particles in microscope imagery.

---

## 🔍 Overview

This repository focuses on the **detection, classification, and counting** of spherical **PMMA microplastics** in high-speed microscopy videos using **YOLOv5**.  
It also extracts **quantitative focus and morphology features** to assess detection quality and particle characteristics.

### Detected classes
- `PMMA10`  
- `PMMA20`

### Core workflow
1. **Dataset preparation** (YOLO format, 70/20/10 split)  
2. **YOLOv5 training** for microparticle detection  
3. **Inference and feature extraction**  
4. **Feature-based 3D visualization and counting**

---

## 🧠 Background

Traditional thresholding and contour-based methods (OpenCV) suffer from focus and lighting sensitivity.  
This YOLOv5-based approach achieves **robust real-time detection** even under variable microscope illumination and motion blur.  
Post-processing evaluates detected particles based on:
- Otsu-binarized area (µm²)
- Laplacian variance (focus sharpness)
- Histogram similarity (Chi-square distance)

---

## 📁 Project Structure

```
Microparticle-Detection-YOLOv5/
│
├── data/
│   ├── images/{train,val,test}/   # Microscopy frames
│   ├── labels/{train,val,test}/   # YOLO txt annotations
│   └── data.yaml                  # Dataset configuration
│
├── yolov5/                        # Ultralytics YOLOv5 (submodule)
│
├── scripts/
│   ├── setup_yolov5.sh            # Setup YOLOv5 + dependencies
│   ├── split_dataset.py           # Dataset split helper
│   ├── train_yolov5.py            # Train YOLOv5s model
│   ├── make_reference_hist.py     # Reference histogram builder
│   ├── infer_video.py             # Detection + feature extraction
│   └── plot_3d_features.py        # 3D feature visualization
│
├── src/
│   ├── postproc.py                # Otsu area, Laplacian, chi-square
│   ├── viz.py                     # Bounding box drawing
│   └── utils.py                   # Utility helpers
│
├── reports/                       # Outputs: CSVs, plots, metrics
├── yolov5_hyp_no_rotate.yaml      # Custom augmentations (no rotation)
├── requirements.txt
├── environment.yml
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/martinbadrous/Microparticle-Detection-YOLOv5.git
cd Microparticle-Detection-YOLOv5
bash scripts/setup_yolov5.sh
```

Optional virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

---

## 🧩 Dataset Setup

Organize data in YOLO format:
```
data/
├── images/train/
├── images/val/
├── images/test/
├── labels/train/
├── labels/val/
└── labels/test/
```

Each `.txt` file should contain YOLO labels:
```
<class_id> <x_center> <y_center> <width> <height>
```
where:
```
0 → PMMA10
1 → PMMA20
```

If you have a single folder of images:
```bash
python scripts/split_dataset.py --images_dir data/images_all --out_root data
```

---

## 🚀 Training the Model

Train YOLOv5s for 50 epochs:
```bash
python scripts/train_yolov5.py
```

Training results (metrics, confusion matrices, precision-recall curves) are stored under:
```
runs/train/mp_yolov5/
```

---

## 🧾 Post-Processing Workflow

### 1️⃣ Create Reference Histogram
```bash
python scripts/make_reference_hist.py --crop path/to/best_particle.png
```

### 2️⃣ Run Inference on Video
```bash
python scripts/infer_video.py   --weights runs/train/mp_yolov5/weights/best.pt   --video path/to/video.mp4   --ref_hist_json reference_hist.json   --um2_per_px 0.30
```

Produces:
```
reports/features_3d.csv
```

### 3️⃣ Generate 3D Feature Plot
```bash
python scripts/plot_3d_features.py
```

Output: `reports/features_3d.png`  
Axes correspond to:
- **X:** Otsu area (px)
- **Y:** Laplacian variance (focus)
- **Z:** Chi-square histogram distance

---

## 🧮 Measurement Units

| Metric | Unit / Note |
|:-------|:-------------|
| Pixel area | ≥ 500 px (min valid particle size) |
| Conversion | 0.30 µm²/pixel |
| Objective | 20× microscope |
| Frame rate | ~30 fps |

---

## 📊 Evaluation Metrics

| Metric | PMMA10 | PMMA20 | Mean |
|:--------|:--------|:--------|:------|
| Precision | 0.80 | 0.79 | 0.795 |
| Recall | 0.88 | 0.91 | 0.895 |
| mAP@0.5 | 0.97 | 0.97 | **0.97** |

---

## 🧠 Model Details

| Parameter | Value |
|:-----------|:--------|
| Model | YOLOv5s |
| Epochs | 50 |
| Image size | 640×640 |
| IoU threshold | 0.5 |
| Rotation | Disabled |
| Augmentations | Gaussian noise, blur, photometric (brightness/contrast/saturation) |

---

## 🎓 Citation

> **Badrous, M.** (2021). *High-Speed Microparticle Detection and Tracking.*  
> ViBOT Master’s Program, Bourgogne University.

---

## 🧭 License

This repository is distributed under the **MIT License** — see `LICENSE` for details.

---

## 💡 Future Work

- Integrate tracking and temporal counting  
- Deploy as a Hugging Face interactive demo  
- Train on larger, more diverse microplastic datasets  
- Explore YOLOv8 or RT-DETR for real-time edge deployment

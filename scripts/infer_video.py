import argparse, os, json, cv2 as cv, torch, pandas as pd, numpy as np
from src.postproc import otsu_area_px, laplacian_variance, hist_256, enhance_contrast_pointwise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--classes", nargs="+", type=int, default=None, help="e.g., 0 1")
    ap.add_argument("--min_area_px", type=int, default=500)
    ap.add_argument("--ref_hist_json", required=True)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--um2_per_px", type=float, default=0.30)
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.ref_hist_json) as f:
        ref = json.load(f)
    ref_alpha = args.alpha if args.alpha is not None else ref.get("alpha", 1.2)
    ref_hist = np.array(ref["hist"], dtype=np.float64)

    # Load YOLOv5
    model = torch.hub.load("ultralytics/yolov5", "custom", path=args.weights)
    if args.classes is not None:
        model.classes = args.classes
    model.conf = args.conf

    cap = cv.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    records = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        results = model(frame, size=args.imgsz)
        df = results.pandas().xyxy[0]
        for _,row in df.iterrows():
            x1,y1,x2,y2 = map(int, [row.xmin,row.ymin,row.xmax,row.ymax])
            cls_name = row.get("name", str(row.get("class", "")))
            conf=float(row["confidence"])
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                continue
            gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
            area_px, thr = otsu_area_px(gray)
            if area_px < args.min_area_px:
                continue
            blur_var = laplacian_variance(gray)
            gray_ce = enhance_contrast_pointwise(gray, alpha=ref_alpha)
            h = hist_256(gray_ce, normalize=True)
            chi2 = float(((h - ref_hist)**2 / (h + ref_hist + 1e-8)).sum())
            area_um2 = area_px * args.um2_per_px if args.um2_per_px else None
            records.append({
                "frame": frame_idx,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": x2-x1, "h": y2-y1,
                "cls": cls_name, "conf": conf,
                "area_px": area_px, "area_um2": area_um2,
                "blur_var": blur_var, "chi2": chi2
            })
    cap.release()

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(args.out_dir, "features_3d.csv")
    df.to_csv(csv_path, index=False)
    print("Wrote", csv_path)

if __name__ == "__main__":
    main()

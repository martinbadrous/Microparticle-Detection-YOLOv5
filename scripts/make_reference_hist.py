import argparse, cv2 as cv, json
import numpy as np

def hist_256(img_gray, normalize=True):
    h = cv.calcHist([img_gray],[0],None,[256],[0,256]).flatten()
    if normalize:
        s = h.sum() + 1e-8
        h = h / s
    return h

def enhance_contrast_pointwise(img_gray, alpha=1.2, beta=0):
    out = cv.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", required=True, help="Path to an 'optimal' particle crop (in-focus)")
    ap.add_argument("--out", default="reference_hist.json")
    ap.add_argument("--alpha", type=float, default=1.2)
    args = ap.parse_args()

    img = cv.imread(args.crop, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.crop)
    img = enhance_contrast_pointwise(img, alpha=args.alpha)
    h = hist_256(img, normalize=True)

    with open(args.out, "w") as f:
        json.dump({"alpha": args.alpha, "hist": h.tolist()}, f)
    print("Saved", args.out)

if __name__ == "__main__":
    main()

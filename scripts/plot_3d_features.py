import argparse, pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/features_3d.csv")
    ap.add_argument("--by", default="cls")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k,grp in df.groupby(args.by):
        ax.scatter(grp["area_px"], grp["blur_var"], grp["chi2"], label=str(k), s=10)

    ax.set_xlabel("Area (px)")
    ax.set_ylabel("Blur (Laplacian variance)")
    ax.set_zlabel("Chi-square distance")
    ax.legend()
    plt.tight_layout()
    out = "reports/features_3d.png"
    plt.savefig(out, dpi=200)
    print("Saved", out)

if __name__ == "__main__":
    main()

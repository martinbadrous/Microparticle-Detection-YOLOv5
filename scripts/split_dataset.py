import argparse, os, random, shutil, glob
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images_dir', default='data/images_all', help='Where all raw images live')
    ap.add_argument('--labels_dir', default='data/labels_all', help='Where all YOLO txt labels live (optional)')
    ap.add_argument('--out_root', default='data', help='Output root with images/{train,val,test}')
    ap.add_argument('--train', type=float, default=0.7)
    ap.add_argument('--val', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    imgs = []
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'):
        imgs.extend(glob.glob(os.path.join(args.images_dir, ext)))
    imgs = sorted(imgs)
    assert len(imgs)>0, f"No images found in {args.images_dir}"

    n = len(imgs)
    n_train = int(n*args.train)
    n_val = int(n*args.val)
    indices = list(range(n))
    random.shuffle(indices)
    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train:n_train+n_val])
    test_idx = set(indices[n_train+n_val:])

    def place(split, idx_set):
        img_dst = Path(args.out_root)/"images"/split
        lbl_dst = Path(args.out_root)/"labels"/split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(imgs):
            if i in idx_set:
                base = os.path.splitext(os.path.basename(p))[0]
                shutil.copy2(p, img_dst/base+Path(p).suffix)
                if os.path.isdir(args.labels_dir):
                    lbl = os.path.join(args.labels_dir, base + ".txt")
                    if os.path.exists(lbl):
                        shutil.copy2(lbl, lbl_dst/base+".txt")
    place("train", train_idx)
    place("val", val_idx)
    place("test", test_idx)
    print("Done.")

if __name__ == "__main__":
    main()

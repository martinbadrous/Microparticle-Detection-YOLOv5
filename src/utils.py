import os, glob

def list_images(folder):
    paths=[]
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff'):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(paths)

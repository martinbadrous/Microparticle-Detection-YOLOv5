import cv2 as cv
import numpy as np

def otsu_area_px(img_gray):
    img_blur = cv.GaussianBlur(img_gray, (3,3), 0)
    _, thr = cv.threshold(img_blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    if np.mean(thr) < 127:
        thr = cv.bitwise_not(thr)
    area = int((thr == 255).sum())
    return area, thr

def laplacian_variance(img_gray):
    lap = cv.Laplacian(img_gray, cv.CV_64F, ksize=3)
    return float(lap.var())

def hist_256(img_gray, normalize=True):
    h = cv.calcHist([img_gray],[0],None,[256],[0,256]).flatten()
    if normalize:
        s = h.sum() + 1e-8
        h = h / s
    return h

def chi_square_distance(h1, h2, eps=1e-8):
    num = (h1 - h2) ** 2
    den = h1 + h2 + eps
    return float((num / den).sum())

def enhance_contrast_pointwise(img_gray, alpha=1.2, beta=0):
    out = cv.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
    return out

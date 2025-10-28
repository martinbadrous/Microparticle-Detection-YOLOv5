import cv2 as cv

def draw_box(img, x1,y1,x2,y2, label=None):
    cv.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    if label:
        cv.putText(img, label, (x1, max(0,y1-5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv.LINE_AA)
    return img

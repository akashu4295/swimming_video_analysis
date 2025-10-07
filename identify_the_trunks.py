import cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

img_path = Path("Frames/frame101.jpg")
img = cv2.imread(str(img_path))
orig = img.copy()
h, w = img.shape[:2]

def detect_trunks(image):
    hh, ww = image.shape[:2]
    crop = image[int(hh*0.35):int(hh*0.65), int(ww*0.15):int(ww*0.85)].copy()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_dark = cv2.inRange(hsv, [0, 30, 0], [180, 255, 80])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask_clean = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_bbox = None
    max_area = 0
    for cnt in contours:
        x,y,wwc,hhc = cv2.boundingRect(cnt)
        area = wwc*hhc
        if area > max_area and area > 100:
            ar = wwc/float(hhc)
            if ar > 1.0:
                max_area = area
                # shift coordinates back to the full image
                best_bbox = (x + int(ww*0.15), y + int(hh*0.35), wwc, hhc)
    return best_bbox, mask_clean, crop

trunk_bbox, trunk_mask, trunk_crop = detect_trunks(orig)

vis = orig.copy()
if trunk_bbox:
    x,y,wwc,hhc = trunk_bbox
    cv2.rectangle(vis, (x,y), (x+wwc, y+hhc), (0,255,0), 2)
    cv2.putText(vis, "Trunks", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

out_path = Path("output_detection.png")
cv2.imwrite(str(out_path), vis)

print("Trunk bbox:", trunk_bbox)


# # frames is list of file paths sorted by time, fps is known
# fps = 60.0  # e.g. adjust to your camera
# pixels_per_cm = (mean_pixel_per_band/35.0) if mean_pixel_per_band else None  # from band detection

# def trunk_center_from_frame(frame_path):
#     # reuse the detect_trunks function above to get bbox and return center x
#     img = cv2.imread(frame_path)
#     bbox, _, _ = detect_trunks(img)
#     if bbox:
#         x,y,w,h = bbox
#         cx = x + w//2
#         cy = y + h//2
#         return (cx, cy)
#     return None

# centers = []
# for p in frames:
#     c = trunk_center_from_frame(p)
#     centers.append(c)

# velocities = []
# for i in range(1, len(centers)):
#     if centers[i] and centers[i-1]:
#         dx_px = centers[i][0] - centers[i-1][0]  # horizontal displacement
#         dx_cm = dx_px / pixels_per_cm
#         v = dx_cm * fps  # cm/s for consecutive frames
#         velocities.append(v)

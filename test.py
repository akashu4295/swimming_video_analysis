import cv2
import numpy as np
import glob
import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt

# Plotting functions for processed data
def plot_velocity(df):
    """Plot swimmer velocity (m/s) across frames."""
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df["velocity_mps"], marker='o', linewidth=1.5)
    plt.title("Swimmer Velocity Across Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_position(df):
    """Plot swimmer position (pixels) across frames."""
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df["position_px"], marker='o', color='orange', linewidth=1.5)
    plt.title("Swimmer Position Across Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Position (pixels)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def plot_scale(df):
    """Plot pixel-to-meter scale across frames."""
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df["pixel_to_meter"], marker='o', color='green', linewidth=1.5)
    plt.title("Pixel-to-Meter Scale Across Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Pixels per Meter")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_data(df):
    plot_velocity(df)
    plot_position(df)
    plot_scale(df)

# Kalman Filter for 1D position and velocity
def init_kalman(dt=1.0, process_var=1e-2, meas_var=1e-1):
    A = np.array([[1, dt],
                  [0, 1]])
    H = np.array([[1, 0]])
    Q = process_var * np.array([[dt**4/4, dt**3/2],
                                 [dt**3/2, dt**2]])
    R = np.array([[meas_var]])
    x = np.array([[0.0], [0.0]])
    P = np.eye(2)
    return dict(A=A, H=H, Q=Q, R=R, x=x, P=P)

def kalman_predict(kf):
    kf['x'] = kf['A'] @ kf['x']
    kf['P'] = kf['A'] @ kf['P'] @ kf['A'].T + kf['Q']
    return kf['x'][0, 0]

def kalman_update(kf, z, meas_var=None):
    if meas_var is not None:
        kf['R'] = np.array([[meas_var]])
    S = kf['H'] @ kf['P'] @ kf['H'].T + kf['R']
    K = kf['P'] @ kf['H'].T @ np.linalg.inv(S)
    y = np.array([[z]]) - kf['H'] @ kf['x']
    kf['x'] = kf['x'] + K @ y
    kf['P'] = (np.eye(2) - K @ kf['H']) @ kf['P']
    return kf['x'][0, 0]

# Lane-band detection 
def detect_lane_bands(img, y_min=0, y_max=None):
    """
    Detect lane bands in the image within a vertical range.
    img : input BGR image
    y_min : minimum y-coordinate (top) to search
    y_max : maximum y-coordinate (bottom) to search. If None, uses image height
    """
    if y_max is None:
        y_max = img.shape[0]  # full image height

    # Preprocessing with Gaussian smoothening preseving edges
    blurred = cv2.bilateralFilter(img, 9, 75, 75)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Color masks
    lower_yellow, upper_yellow = np.array([20, 20, 20]), np.array([35, 255, 255])
    lower_green, upper_green = np.array([35, 20, 20]), np.array([85, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(mask_yellow, mask_green)

    # Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids, boxes = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter by reasonable width and height
        if 90 <= w <= 160 and h > 20:
            # Filter by vertical range
            if y_min <= y <= y_max:
                cx, cy = x + w // 2, y + h // 2
                centroids.append((cx, cy))
                boxes.append((x, y, w, h))

    # Sort by vertical position
    centroids = sorted(centroids, key=lambda c: c[1])
    boxes = sorted(boxes, key=lambda b: b[1])

    return centroids, boxes

# Helper function to lane band detection
def most_common_centroids(centroids, tol=10):
    """
    Identify the Most repeated centroids based on the y coordinate
    within a tolerance of 10 pixels
    Args:
        centroids (x,y): List of centroids
        tol (int, optional): tolerance for y coordinates. Defaults to 10.
    Returns:
        returns : Average of the most common centroids (x_avg, y_avg)
    """
    binned = [round(cy / tol) * tol for _, cy in centroids]
    counts = Counter(binned)
    most_common_y = counts.most_common(1)[0][0]
    return np.average([c for c in centroids if abs(c[1] - most_common_y) <= tol], axis=0)

# Swimmer line detection
def detect_swimmer_lines(img, prev_x=None, search_radius=80):
    """
    Detects close to verticall lines in the image using HoughLinesP()
    Args:
        img (_type_): RGB image
        prev_x (_type_, optional): Used for prediction if previous image data is known. Defaults to None.
        search_radius (int, optional): Number of pixels in radius to search around for based on the previous position. Defaults to 80.
    Returns:
        returns: candidates for the x locations of the new set of line and confidence values
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape
    # Search space or Region of Interest
    x_min, x_max = (int(w*0.3), int(w*0.7))
    y_min, y_max = (int(h*0.4), int(h*0.7))
    roi = edges[y_min:y_max, x_min:x_max]
    
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=20, minLineLength=25, maxLineGap=10)
    x_candidates, confidences = [], []
    
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) > 80:
                x_global = 0.5*(x1 + x2) + x_min
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                confidence = length
                if prev_x is None or abs(x_global - prev_x) < search_radius:
                    x_candidates.append(x_global)
                    confidences.append(confidence)
    
    if len(x_candidates) == 0:
        return None, 0.0
    best_idx = np.argmax(confidences)
    return x_candidates[best_idx], confidences[best_idx]

# Main Code
# ------------------------------
frame_dir = "FRAMES"
out_dir = "output_frames"
os.makedirs(out_dir, exist_ok=True)

frames = sorted(glob.glob(os.path.join(frame_dir, "frame*.jpg")))
if len(frames) == 0:
    print("No frames found.")
    exit(1)

fps = 30  # <-- adjust if known
dt = 1.0 / fps
kf = init_kalman(dt=dt)

prev_x = None
pixel_to_meter = None
positions, velocities, frame_ids, confidences, scales = [], [], [], [], []
lane_spacing_list = []

for i, frame_path in enumerate(frames):
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    draw = frame.copy()

    # ---- Lane-band detection ----
    centroids, boxes = detect_lane_bands(frame, 170, 400)
    centroid_avg = most_common_centroids(centroids, tol=15) if centroids else None

    if centroid_avg is not None:
        ys = np.array([c[1] for c in centroids])
        indices = np.where((ys > centroid_avg[1] - 10) & (ys < centroid_avg[1] + 10))[0]
        centroids = [centroids[j] for j in indices]
        boxes = [boxes[j] for j in indices]
        
    if len(centroids) >= 2:
        spacings = [abs(centroids[j+1][0] - centroids[j][0]) for j in range(len(centroids)-1)]
        median_spacing = np.median(spacings)
        lane_spacing_list.append(median_spacing)
        if len(lane_spacing_list) > 10:
            lane_spacing_list.pop(0)
        pixel_to_meter = np.median(lane_spacing_list) / 0.7  # px per meter
        # if (pixel_to_meter < 15) or (pixel_to_meter > 25):
        #     pixel_to_meter = 19

    # draw lane boxes
    for (x, y, w, h) in boxes:
        cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ---- Swimmer line detection ----
    meas_x, conf = detect_swimmer_lines(frame, prev_x, search_radius=120)
    pred_x = kalman_predict(kf)
    if i == 0:
        if meas_x is not None:
            kf['x'][0,0] = meas_x
            prev_x = meas_x
        continue

    if meas_x is not None:
        meas_var = 1.0 / (conf + 1e-6)
        est_x = kalman_update(kf, meas_x, meas_var=meas_var)
    else:
        est_x = pred_x

    vx = kf['x'][1,0]
    vx_mps = vx / pixel_to_meter if pixel_to_meter else np.nan

    positions.append(est_x)
    velocities.append(vx_mps)
    frame_ids.append(os.path.basename(frame_path))
    confidences.append(conf)
    scales.append(pixel_to_meter)

    # ---- Draw predictions ----
    h, w = frame.shape[:2]
    y1, y2 = int(h*0.3), int(h*0.8)
    if meas_x is not None:
        cv2.line(draw, (int(meas_x), y1), (int(meas_x), y2), (0,0,255), 2)  # red measured
    cv2.line(draw, (int(est_x), y1), (int(est_x), y2), (255,0,0), 1, lineType=cv2.LINE_AA)  # blue estimated/pred
    cv2.putText(draw, f"Frame {i+1}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(draw, f"vx = {vx_mps:.2f} m/s", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    out_path = os.path.join(out_dir, f"out_{i:04d}.jpg")
    cv2.imwrite(out_path, draw)
    prev_x = est_x

# ---- Save CSV ----
df = pd.DataFrame({
    "frame": frame_ids,
    "position_px": positions,
    "velocity_mps": velocities,
    "confidence": confidences,
    "pixel_to_meter": scales
})
df.to_csv("swimmer_velocity.csv", index=False)
print("Done. Annotated frames in 'output_frames/' and CSV saved as swimmer_velocity.csv")
plot_data(df)
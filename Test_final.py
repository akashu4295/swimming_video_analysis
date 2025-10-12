import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from statistics import median
import csv

# ---------------------- CONFIGURATION ----------------------
frames_dir = "FRAMES"
frame_prefix = "frame"
frame_start, frame_end = 101, 200
fps = 30.0  # frames per second
band_length_m = 0.35  # 35 cm per color band
search_window = 60     # initial tracking window (pixels)
csv_filename = "swimmer_velocity_data.csv"
# -----------------------------------------------------------


def detect_lane_bands(img):
    """Detect lane band center y-positions (green/yellow)."""
    blurred = cv2.bilateralFilter(img, 9, 75, 75)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_yellow, upper_yellow = np.array([20, 20, 20]), np.array([35, 255, 255])
    lower_green, upper_green = np.array([35, 20, 20]), np.array([85, 255, 255])

    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(mask_y, mask_g)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y_positions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:
            y_positions.append(y + h // 2)
    return sorted(y_positions) if y_positions else None


def pixel_to_meter_conversion(band_positions):
    """Estimate pixel-to-meter ratio using median spacing of color bands."""
    if not band_positions or len(band_positions) < 2:
        return None
    pixel_dist = np.median(np.diff(sorted(band_positions)))
    return band_length_m / pixel_dist


def detect_vertical_lines(img, prev_xs=None, search_window=60):
    """
    Detect near-vertical lines using Hough Transform.
    If prev_xs provided, search only nearby regions.
    Returns list of x-positions (in full-image coordinates).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = img.shape[:2]

    # Central ROI to avoid lens distortions
    cx1, cx2 = int(w * 0.25), int(w * 0.75)
    cy1, cy2 = int(h * 0.3), int(h * 0.7)
    roi = edges[cy1:cy2, cx1:cx2]

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=20,
                            minLineLength=15, maxLineGap=10)

    x_positions = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) > 80:  # near vertical
                x_full = (x1 + x2) / 2 + cx1
                # Restrict to nearby positions if previous known
                if prev_xs is not None:
                    if any(abs(x_full - px) < search_window for px in prev_xs):
                        x_positions.append(x_full)
                else:
                    x_positions.append(x_full)
    return x_positions


# ---------------------- MAIN PROCESS ----------------------

positions_all = []   # list of lists of x-positions per frame
velocities = []      # framewise median velocities
pixel_to_meter = None
prev_xs = None

for frame_no in range(frame_start, frame_end + 1):
    filename = os.path.join(frames_dir, f"{frame_prefix}{frame_no}.jpg")
    img = cv2.imread(filename)
    if img is None:
        print(f"⚠️ Could not read {filename}")
        positions_all.append([])
        continue

    # Detect lane bands every 10 frames for scaling
    if frame_no == frame_start or (frame_no - frame_start) % 10 == 0:
        band_positions = detect_lane_bands(img)
        if band_positions is not None:
            new_scale = pixel_to_meter_conversion(band_positions)
            if new_scale:
                pixel_to_meter = new_scale

    # Detect vertical lines (track from previous)
    x_positions = detect_vertical_lines(img, prev_xs, search_window)
    if not x_positions and prev_xs:
        # Expand search window if lost
        x_positions = detect_vertical_lines(img, prev_xs, search_window * 2)
    positions_all.append(x_positions)
    prev_xs = x_positions if x_positions else prev_xs  # keep last known


# ---------------------- VELOCITY CALCULATION ----------------------

results = []
for i in range(1, len(positions_all)):
    xs_prev = positions_all[i - 1]
    xs_curr = positions_all[i]
    frame_pair = (frame_start + i - 1, frame_start + i)

    if not xs_prev or not xs_curr or pixel_to_meter is None:
        velocities.append(None)
        results.append([frame_pair[1], len(xs_prev), len(xs_curr), None])
        continue

    v_candidates = []
    for xp in xs_prev:
        for xc in xs_curr:
            dx = (xc - xp) * pixel_to_meter
            dt = 1.0 / fps
            v_candidates.append(dx / dt)

    if v_candidates:
        v_arr = np.array(v_candidates)
        # Outlier rejection: 10–90 percentile range
        low, high = np.percentile(v_arr, [10, 90])
        v_filtered = [v for v in v_arr if low <= v <= high]
        v_med = median(v_filtered) if v_filtered else median(v_arr)
        velocities.append(v_med)
        results.append([frame_pair[1], len(xs_prev), len(xs_curr), v_med])
    else:
        velocities.append(None)
        results.append([frame_pair[1], len(xs_prev), len(xs_curr), None])


# ---------------------- SAVE RESULTS TO CSV ----------------------

with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_number", "num_lines_prev", "num_lines_curr", "median_velocity_m_per_s"])
    for row in results:
        writer.writerow(row)

print(f"\n✅ Velocity data saved to '{csv_filename}'.")


# ---------------------- PLOT RESULTS ----------------------

times = np.arange(frame_start, frame_end) / fps
valid_vels = [v if v is not None else np.nan for v in velocities]

plt.figure(figsize=(10, 5))
plt.plot(times, valid_vels, 'o-', label="Median velocity (m/s)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Swimmer Velocity Across Frames (Median of Candidate Lines)")
plt.legend()
plt.grid(True)
plt.show()

# Summary printout
print("\n--- Framewise Velocity Summary ---")
for i, v in enumerate(velocities):
    if v is not None:
        print(f"Frame {frame_start + i} → {frame_start + i + 1}: {v:.3f} m/s")
    else:
        print(f"Frame {frame_start + i} → {frame_start + i + 1}: [no data]")
output = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 20 and h > 20:  # filter out noise
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite("lane_bands_detected_green_yellow.jpg", output)

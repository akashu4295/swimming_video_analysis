import cv2
import numpy as np

# Load image
img = cv2.imread("Frames/frame107.jpg")

# --- Blur / Smoothing Step ---
blurred = cv2.bilateralFilter(img, 9, 75, 75) # Bilateral filter (good for underwater, preserves edges)
# blurred = cv2.GaussianBlur(img, (7, 7), 0)    # Gaussian blur
# blurred = cv2.medianBlur(img, 5)              # Median blur (good for salt-and-pepper noise)

# Convert to HSV
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# Define color ranges for red and yellow bands
# Red (can have two ranges due to HSV wrap-around)
lower_red1 = np.array([0, 20, 20])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 20, 20])
upper_red2 = np.array([180, 255, 255])
lower_yellow = np.array([20, 20, 20])
upper_yellow = np.array([35, 255, 255])
lower_green = np.array([35, 20, 20])
upper_green = np.array([85, 255, 255])

# Create masks
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_green_and_yellow = cv2.bitwise_or(mask_green, mask_yellow)

# Mask Red, identify contours and draw bounding boxes
mask = mask_red
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 20 and h > 20:  # filter out noise
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite("lane_bands_detected_red.jpg", output)

mask = mask_yellow
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 20 and h > 20:  # filter out noise
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite("lane_bands_detected_yellow.jpg", output)

mask = mask_green
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 20 and h > 20:  # filter out noise
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite("lane_bands_detected_green.jpg", output)

mask = mask_green_and_yellow
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 20 and h > 20:  # filter out noise
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite("lane_bands_detected_green_and_yellow.jpg", output)


# grayscaling
# Read the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display the image
cv2.imshow('Grayscale Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the grayscale image
cv2.imwrite('grayscale_image.jpg', gray)

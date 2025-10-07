import cv2
vid = cv2.VideoCapture("The worst stroke.mp4") # Open video file

count, success = 0, True
while success:
    success, image = vid.read() # Read frame
    if success: 
        cv2.imwrite(f"Frames/frame{count}.jpg", image) # Save frame
        count += 1

vid.release()
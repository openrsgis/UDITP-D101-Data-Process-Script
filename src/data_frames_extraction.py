import cv2
import os

# Video file path
video_path = "VID_20241111_143943.mp4"
frames_dir = "20241111_143943/frames"

cap = cv2.VideoCapture(video_path)

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1


cap.release()

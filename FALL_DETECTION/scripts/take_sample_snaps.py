import os
import cv2
import time
from datetime import datetime

# RTSP Stream URL
RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# Open the video stream
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# Check if the stream opened successfully
if not cap.isOpened():
    print("❌ Error: Unable to open RTSP stream")
    exit()

# Directory to save snapshots
SNAPSHOT_DIR = "test_snaps"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Time interval for snapshots (in seconds)
SNAPSHOT_INTERVAL = 1  # Capture every 5 seconds
last_snapshot_time = time.time()

# Function to save a snapshot
def save_fall_snapshot(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SNAPSHOT_DIR}/sitting_snap_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Snapshot saved: {filename}")

# Read and display frames from RTSP stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Warning: Failed to grab frame, retrying...")
        continue

    # Display the frame (optional)
    cv2.imshow("RTSP Stream", frame)

    # Capture a snapshot at defined intervals
    current_time = time.time()
    if current_time - last_snapshot_time >= SNAPSHOT_INTERVAL:
        save_fall_snapshot(frame)
        last_snapshot_time = current_time  # Update last snapshot time

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

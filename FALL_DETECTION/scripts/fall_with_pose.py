import cv2
import torch
import numpy as np
import cvzone
from ultralytics import YOLO
import os

# Load YOLO-Pose Model
pose_model = YOLO("yolov8n-pose.pt")

# Thresholds
ANGLE_THRESHOLD = 40  # Angle close to 90° means fallen
Y_MOVEMENT_THRESHOLD = 50  # Minimum downward movement to be considered a fall
previous_keypoints = {}

# Function to calculate angle between 3 points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Function to detect fall based on pose
def detect_fall_with_pose(keypoints, person_id):
    global previous_keypoints

    if keypoints.shape[0] < 13:  # Ensure at least 13 keypoints are detected
        return False

    # Extract keypoints
    nose, left_shoulder, right_shoulder = keypoints[0], keypoints[5], keypoints[6]
    left_hip, right_hip = keypoints[11], keypoints[12]

    # Calculate torso angle
    torso_angle = calculate_angle(left_shoulder, left_hip, right_hip)

    # Track vertical movement
    prev_y = previous_keypoints.get(person_id, left_hip)[1]
    y_movement = abs(left_hip[1] - prev_y)
    previous_keypoints[person_id] = left_hip  # Update tracking

    return torso_angle < ANGLE_THRESHOLD and y_movement > Y_MOVEMENT_THRESHOLD

# Video Stream
RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
# cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/hafeefa2_fall.mp4")
# cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/fall2.mp4")
cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose_model(frame)

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints else []
        if len(keypoints) == 0:
            continue  # Skip if no detections

        for person_id, keypoint in enumerate(keypoints):
            fall_detected = detect_fall_with_pose(keypoint, person_id)

            # Draw Keypoints
            for (x, y) in keypoint:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Display Fall Alert
            if fall_detected:
                cvzone.putTextRect(frame, "FALL DETECTED!", (50, 50), scale=2, thickness=3, colorR=(0, 0, 255))
                cv2.rectangle(frame, (50, 50), (300, 100), (0, 0, 255), -1)

    cv2.imshow("Fall Detection with YOLO-Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

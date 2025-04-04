import cv2
import os
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import threading

# Load YOLO model
model = YOLO('/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/scripts/runs/detect/train_dataset/weights/best.pt')

# Allowed object classes
ALLOWED_CLASSES = {"Fall Detected", "Walking","Sitting"}

# RTSP Camera URL
RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'

# # Set FFmpeg options
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp;'

# # Open the RTSP stream    
cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)
# cap=cv2.VideoCapture(0)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to process the frame
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    pose_results = pose.process(rgb_frame)
    
    detected_label = "unknown"

    if pose_results.pose_landmarks:
        detected_label = classify_pose(pose_results.pose_landmarks.landmark)
        
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = model.names[class_id]
            
            if label in ALLOWED_CLASSES and confidence > 0.5:
                detected_label = label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display detected pose label
    cv2.putText(frame, detected_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return frame

# Function to classify body pose
def classify_pose(landmarks):
    hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    knee_y = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2
    foot_y = (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y) / 2
    head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    
    if head_y > hip_y and hip_y > knee_y:
        return "Fall Detected"
    elif hip_y < knee_y and knee_y < foot_y:
        return "Walking"
    else:
        return "Sitting"
    

# Run the detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    frame = process_frame(frame)
    
    cv2.imshow("YOLO + MediaPipe Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

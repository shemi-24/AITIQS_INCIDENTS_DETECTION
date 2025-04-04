

import cv2
import os
import mediapipe as mp
from ultralytics import YOLO

# Load YOLO model
model = YOLO('/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/scripts/runs/detect/train_weapon/weights/best.pt')

# Define Allowed Object Classes
ALLOWED_CLASSES = {"pistol", "knife"}

# # RTSP Camera URL
# RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'

# # Set FFmpeg options
# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp;timeout;5000000'

# # Open the RTSP Stream
# cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
# cap=cv2.VideoCapture('/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/hefeefa_weapon.mp4')


RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/401'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
# cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/hafeefa2_fall.mp4")
# cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/fall2.mp4")
cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO model for object detection
    results = model(frame)

    # Convert BGR to RGB (for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID
            label = model.names[class_id]  # Object label (e.g., "pistol", "knife")

            # Filter for only "pistol" and "knife"
            if label in ALLOWED_CLASSES and confidence > 0.7:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Process Hand Tracking with MediaPipe
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # Draw hand landmarks

    # Show Output Frame
    cv2.imshow("YOLOv10 Detection - Only Pistol & Knife", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

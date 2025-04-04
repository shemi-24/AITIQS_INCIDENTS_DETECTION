import cv2
import numpy as np
import os
from ultralytics import YOLO
import face_recognition
import threading
import pickle

# Locks and shared variables
lock = threading.Lock()
face_results = []
fall_results = []
frame = None
running = True

# Load face encodings
FACE_ENCODINGS_PATH = "/Users/zoftcares/StudioProjects/fall_detection_system/model/face_encodings.pkl"
if not os.path.exists(FACE_ENCODINGS_PATH):
    raise FileNotFoundError(f"Face encodings file '{FACE_ENCODINGS_PATH}' not found.")

with open(FACE_ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)
known_encodings, known_names = data["encodings"], data["names"]

# Load YOLO Pose model
MODEL_PATH = "yolov8n-pose.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
model = YOLO(MODEL_PATH)  # Load YOLOv8 Pose model

# Open RTSP stream
RTSP_URL = "rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/401"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/fall_detection_system/hafeefa2_fall.mp4")

if not cap.isOpened():
    raise ConnectionError("Failed to open RTSP stream. Check URL and network connection.")

# Face Recognition Thread
def face_recognition_thread():
    global face_results, frame, running

    while running:
        with lock:
            if frame is None:
                continue
            frame_copy = frame.copy()  # Prevent race condition

        rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        print(f"Face Locations: {face_locations}")  # Debugging step

        if not face_locations:
            continue  # No faces detected

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        results = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            results.append((left, top, right, bottom, name))

        with lock:
            face_results = results  # Store results safely

# Start face recognition thread
threading.Thread(target=face_recognition_thread, daemon=True).start()

def process_keypoints(keypoints):
    if len(keypoints) < 11:
        return "Unknown"  # Not enough keypoints

    head_y = keypoints[0][1]  # Head Y position
    hip_y = keypoints[6][1]   # Hip Y position
    foot_y = keypoints[10][1]  # Foot Y position

    height = abs(head_y - foot_y)

    if height > 200:
        return "Standing"
    elif abs(hip_y - foot_y) < 5:
        return "Fallen"
    else:
        return "Sitting"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read frame from RTSP stream.")
        break

    with lock:
        frame_copy = frame.copy()  # Avoid race conditions

    # Run YOLO Pose estimation
    results = model(frame_copy)

    # Draw Face Recognition Results
    with lock:
        for (left, top, right, bottom, detected_person) in face_results:
            cv2.rectangle(frame_copy, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame_copy, detected_person, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []

        if len(keypoints) > 0 and len(keypoints[0]) > 0:
            pose = process_keypoints(keypoints[0])
            print(f"Detected Pose: {pose}")

            # Draw keypoints and label
            for x, y in keypoints[0]:  
                cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Display classification
            cv2.putText(frame_copy, pose, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print("No keypoints detected!")

    # Show frame
    cv2.imshow("YOLO Pose Estimation", frame_copy)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False  # Stop face recognition thread
        break

cap.release()
cv2.destroyAllWindows()

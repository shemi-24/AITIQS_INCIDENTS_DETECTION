import cv2
import torch
import pickle
import face_recognition
import threading
import numpy as np
import json
import os
from ultralytics import YOLO
from app_utils import putTextRect, send_email

# Load YOLO-Pose model
fall_model = YOLO("/Users/zoftcares/StudioProjects/fall_detection_system/scripts/runs/pose/train/weights/best.pt")

# Load face encodings
with open("/Users/zoftcares/StudioProjects/fall_detection_system/model/face_encodings.pkl", "rb") as f:
    face_data = pickle.load(f)
known_face_encodings = face_data["encodings"]
known_face_names = face_data["names"]

# Open video stream (RTSP or webcam)
RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/fall_detection_system/safa_fall.mp4")

# Shared variables for threading
frame = None
running = True
face_results = []
fall_results = []
lock = threading.Lock()
email_sent = False
last_detected_person = "Unknown"

# Face Recognition Thread
def face_recognition_thread():
    global face_results, running, last_detected_person

    while running:
        if frame is None:
            continue

        # Convert to RGB for face recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and encode
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        detected_person = "Unknown"

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            
            if True in matches:
                first_match_index = matches.index(True)
                detected_person = known_face_names[first_match_index]
                last_detected_person = detected_person  # Update last detected person

            results.append((left, top, right, bottom, detected_person))

        with lock:
            face_results = results

# Fall Detection Thread
def fall_detection_thread():
    global fall_results, running

    while running:
        if frame is None:
            continue

        results = fall_model(frame)

        detected_falls = []
        for result in results:
            boxes = result.boxes  

            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                class_name = fall_model.names[class_id]

                detected_falls.append((x1, y1, x2, y2, class_name, confidence))

        with lock:
            fall_results = detected_falls

# Start Threads
threading.Thread(target=face_recognition_thread, daemon=True).start()
threading.Thread(target=fall_detection_thread, daemon=True).start()

# Video Stream Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw Face Recognition Results
    with lock:
        for (left, top, right, bottom, detected_person) in face_results:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, detected_person, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw Fall Detection Results
    with lock:
        for (x1, y1, x2, y2, class_name, confidence) in fall_results:
            if confidence > 0.6:
                color = (0, 255, 0) if class_name == "Walking" else (255, 255, 0) if class_name == "Sitting" else (0, 0, 255)
                label = "Person Walking" if class_name == "Walking" else "Person Sitting" if class_name == "Sitting" else f"Fall Detected ({last_detected_person})"
                
                if class_name == "Fall Detected":
                    print(f"⚠️ Fall detected! Checking conditions for {last_detected_person}...")

                    if last_detected_person != "Unknown" and not email_sent:
                        print(f"📩 Sending email alert for {last_detected_person}...")
                        alert_data = {"alert": f"Fall Detected for {last_detected_person}"}
                        print(json.dumps(alert_data))

                        putTextRect(frame, f"Fall Detected! ({last_detected_person})", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

                        email_body = f"{last_detected_person} has fallen. Immediate attention needed!"
                        # Uncomment to enable email sending
                        # if send_email("admin@example.com", "Fall Detected!", email_body):
                        #     email_sent = True  

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)

    cv2.imshow("Fall Detection System with Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()

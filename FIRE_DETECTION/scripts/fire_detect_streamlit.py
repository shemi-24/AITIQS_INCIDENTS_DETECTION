import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import time
from datetime import datetime

# Load YOLO Model
model = YOLO("/Users/zoftcares/StudioProjects/AI PROJECTS/AITIQS_INCIDENTS_DETECTION/FIRE_DETECTION/best.pt")  # Use your trained model

# Initialize Variables
RTSP_URL = ""
last_detection_time = {}  # Store last detection time
cooldown_time = 900  # 15 minutes

# Streamlit UI
st.title("🔥 Fire & Smoke Detection with AI")
st.subheader("Enter the RTSP Camera URL")

# User Input for RTSP URL
RTSP_URL = st.text_input("RTSP Stream URL", "rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301")

# Start Detection Button
start_detection = st.button("Start Detection")

if start_detection and RTSP_URL:
    st.write("🔍 Connecting to the camera...")

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    # Open Camera Stream
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        st.error("❌ Could not open the RTSP stream. Please check the URL.")
    else:
        st.success("✅ Camera stream started!")

        # Streamlit Video Placeholder
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Lost connection to the camera. Stopping...")
                break
            
            # Run YOLO detection
            results = model(frame)

            # Process Results
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding Box
                    conf = float(box.conf[0])  # Confidence Score
                    cls = int(box.cls[0])  # Class ID

                    # Labels
                    label = "Fire" if cls == 0 else "Smoke"
                    color = (0, 0, 255) if cls == 0 else (255, 255, 0)

                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Cooldown Logic
                    current_time = time.time()
                    if label not in last_detection_time or (current_time - last_detection_time[label] > cooldown_time):
                        st.warning(f"🚨 {label} Detected!")
                        last_detection_time[label] = current_time
            
            # Convert frame for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # Break if the user stops the app
            if not start_detection:
                break

        cap.release()
        cv2.destroyAllWindows()

st.info("🔴 Press 'Stop' on the Streamlit app to stop detection.")

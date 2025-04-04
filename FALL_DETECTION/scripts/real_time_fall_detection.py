

import cv2
import face_recognition
import pickle
import json
import torch
import threading
import numpy as np
from ultralytics import YOLO
from app_utils import putTextRect, send_email
import os
import pandas as pd
import cvzone
import time
from datetime import datetime

import requests



my_file = open("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Load face encodings
with open("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/model/face_encodings.pkl", "rb") as f:
    data = pickle.load(f)
known_encodings, known_names = data["encodings"], data["names"]

# Load YOLO fall detection model
# fall_model = YOLO("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/scripts/runs/detect/train_dataset/weights/best.pt")

fall_model = YOLO("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/scripts/runs/detect/hashir_cctv_fall(26-03-2025)/weights/best.pt")


model = YOLO("yolov10s.pt")  


# LOAD YOLO POSE DETECTION MODEL
# fall_model = YOLO("/Users/zoftcares/StudioProjects/fall_detection_system/scripts/runs/pose/train/weights/best.pt")


# Open RTSP stream
RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
# cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/hafeefa2_fall.mp4")
# cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/fall2.mp4")
cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)
# cap = cv2.VideoCapture(0)

# Shared Variables
frame = None
running = True
face_results = []
fall_results = []
lock = threading.Lock()
previous_y_positions = {}
email_sent = False
last_detected_person = "Unknown"
count=0

# Face Recognition Thread
def face_recognition_thread():
    global face_results, running, last_detected_person

    while running:
        if frame is None:
            continue

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []
        detected_person = "Unknown"

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            
            if True in matches:
                first_match_index = matches.index(True)
                detected_person = known_names[first_match_index]
                last_detected_person = detected_person

            results.append((left, top, right, bottom, detected_person))

        # Store results in thread-safe way
        with lock:
            face_results = results

# Fall Detection Thread
def fall_detection_thread():
    global fall_results, running

    while running:
        if frame is None:
            continue

        # YOLO Fall Detection
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

        # Store fall detection results safely
        with lock:
            fall_results = detected_falls

# Start Threads
threading.Thread(target=face_recognition_thread, daemon=True).start()
threading.Thread(target=fall_detection_thread, daemon=True).start()


# Directory to save snapshots
SNAPSHOT_DIR = "fall_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Initialize detection buffer and cooldown tracker
detection_buffer = []
last_detection_time = {}  # Dictionary to store last detection timestamps per weapon

GRAPHQL_URL = "http://43.205.136.121:9001/graphql"

# Directory to save snapshots
SNAPSHOT_DIR = "weapon_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


HEADERS = {
    "Content-Type": "application/json",
    "x-tenant-id": "zoftcares"
}


# 🔍 Function to fetch the service ID dynamically
def get_service_id():
    query = """
        query Services($input: ServicesFilterInput) {
            services(input: $input) {
                totalCount
                data {
                    id
                    name
                    description
                }
            }
        }
    """
    response = requests.post(GRAPHQL_URL, json={"query": query, "variables": {}}, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        services = data.get("data", {}).get("services", {}).get("data", [])
        if services:
            return services[0]["id"]  # Return the first service ID (modify if needed)
        else:
            print("⚠️ No services found.")
            return None
    else:
        print(f"❌ Failed to fetch services: {response.text}")
        return None
    
    # 🔍 Function to fetch the urgency ID dynamically
def get_urgency_id():
    query = """
        query Urgencies {
            urgencies {
                id
                name
            }
        }
    """
    response = requests.post(GRAPHQL_URL, json={"query": query, "variables": {}}, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        urgencies = data.get("data", {}).get("urgencies", {})
        if urgencies:
            return urgencies[2]["id"]  # Return the first service ID (modify if needed)
        else:
            print("⚠️ No urgencies found.")
            return None
    else:
        print(f"❌ Failed to fetch urgencies: {response.text}")
        return None
    
    
    
 # 🔍 Function to fetch the urgency ID dynamically
def get_users_id():
    query = """
        query Users {
            users {
                totalCount
                users {
                    id
                    tenant_key
                }
            }
        }
    """
    response = requests.post(GRAPHQL_URL, json={"query": query, "variables": {}}, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        users = data.get("data", {}).get("users", {}).get("users",[])
        if users:
            return users[0]["id"]  # Return the first service ID (modify if needed)
        else:
            print("⚠️ No users found.")
            return None
    else:
        print(f"❌ Failed to fetch users: {response.text}")
        return None
    
# 🔍 Function to fetch the priorityId dynamically
def get_priority_id():
    query = """
       query Priorities {
            priorities {
                id
                name
            }
        }
    """
    response = requests.post(GRAPHQL_URL, json={"query": query, "variables": {}}, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        priorities = data.get("data", {}).get("priorities", {})
        if priorities:
            return priorities[0]["id"]  # Return the first service ID (modify if needed)
        else:
            print("⚠️ No priorities found.")
            return None
    else:
        print(f"❌ Failed to fetch priorities: {response.text}")
        return None


# 📸 Function to capture snapshot and send alert
def save_fall_snapshot(frame, service):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SNAPSHOT_DIR}/{service}_fall_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Snapshot saved: {filename}")

    # Fetch the service ID dynamically
    service_id = get_service_id()
    if not service_id:
        print("⚠️ No valid service ID, skipping alert.")
        return
    else:
        print("SERVICE")
        print(service_id);
     
    # Fetch the urgency ID dynamically    
    urgency_id = get_urgency_id()
    if not urgency_id:
        print("⚠️ No valid urgency ID, skipping alert.")
        return
    else:
        print("URGENCY")
        print(urgency_id);
        
    # Fetch the priority ID dynamically    
    priority_id = get_priority_id()
    if not priority_id:
        print("⚠️ No valid priority ID, skipping alert.")
        return
    else:
        print("PRIORITY")
        print(priority_id);
        
        
    # Fetch the user ID dynamically    
    user_id = get_users_id()
    if not priority_id:
        print("⚠️ No valid user ID, skipping alert.")
        return
    else:
        print("USER")
        print(user_id);

    mutation = """
        mutation CreateIncident($input: CreateIncidentInput!) {
            createIncident(input: $input) {
                incident {
                    id
                }
                message
            }
        }
    """
    
    variables = {
        "input": {
            "title": f"{service} detected!!!!!!",
            "incidentTypeId": "1",
            "description": f'{service} is detected',
            "urgencyId": urgency_id,
            "priorityId": priority_id,
            "serviceId": service_id,
            "assignedUserId": user_id
        }
    }
    
    print("REQUESTED BODY");
    print(variables)

    response = requests.post(GRAPHQL_URL, json={"query": mutation, "variables": variables}, headers=HEADERS)
    
    
    print("RESPONSE BODY")
    print(response.content)

    if response.status_code == 200:
        print(f"✅ {service} alert sent successfully!")
    else:
        print(f"❌ Failed to send {service} alert:", response.text)

    
# Dictionary to track previous states of detected persons
person_states = {}

# Video Stream Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw Face Recognition Results
    with lock:
        for (left, top, right, bottom, detected_person) in face_results:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, detected_person, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #Draw Fall Detection Results
    with lock:
        for (x1, y1, x2, y2, class_name, confidence) in fall_results:
            if confidence > 0.7:
                color = (0, 255, 0) if class_name == "Walking" else (255, 255, 0) if class_name == "Sitting" else (0, 0, 255)
                label = "Person Walking" if class_name == "Walking" else "Person Sitting" if class_name == "Sitting" else f"Person Fallen ({last_detected_person})"
                
                if class_name == "Fall Detected":
                    print(f"Fall detected! Checking conditions for {last_detected_person}...")

                    if last_detected_person != "Unknown" and not email_sent:
                        print(f"Fall detected! Sending email alert for {last_detected_person}...")
                        alert_data = {"alert": f"Fall Detected for {last_detected_person}"}
                        print(json.dumps(alert_data))

                        putTextRect(frame, f"Fall Detected! ({last_detected_person})", (x1, y1 - 20), scale=2, thickness=3, colorR=color)

                        email_body = f"{last_detected_person} has fallen. Immediate attention needed!"
                        
                         # Cooldown logic
                        current_time = time.time()
                        if label not in last_detection_time or (current_time - last_detection_time[label] > 900):
                            print(f"🟢 {label} DETECTION CONFIRMED. Saving snapshot...")
                            # save_fall_snapshot(frame, label)  # Call the snapshot function
                            last_detection_time[label] = current_time
                        else:
                            time_left = 900 - (current_time - last_detection_time[label])
                            print(f"⏳ {label} detected again but ignored (Cooldown: {time_left:.2f} seconds left).")
                        
                        # if send_email("hashirkp13@gmail.com", "Fall Detected!", email_body):
                        #     email_sent = True  

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=color)
                
                
    # with lock:
    #     ret,frame = cap.read()
    # count += 1
    # if count % 3 != 0:
    #     continue
    # if not ret:
    #    break
    # # frame = cv2.resize(frame, (1020, 600))

    # results = model(frame)
    # a = results[0].boxes.data
    # px = pd.DataFrame(a).astype("float")
    # list=[]
    # for index, row in px.iterrows():
    #     x1 = int(row[0])
    #     y1 = int(row[1])
    #     x2 = int(row[2])
    #     y2 = int(row[3])
        
    #     d = int(row[5])
    #     c = class_list[d]
    #     # if "person" in c:
    #     cvzone.putTextRect(frame,f'{c}',(x1,y1),1,1)
    #     cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)    


    #     h=y2-y1
    #     w=x2-x1
    #     thresh=h-w
    #     print(thresh) 
    #     if 'person' in c:
    #        if thresh < -20:

    #            cvzone.putTextRect(frame,f'{last_detected_person}{"_fall"}',(x1,y1),1,1)
    #            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
               
    #            # Save snapshot
    #            save_fall_snapshot(frame, last_detected_person)
               
    #        else:
    #            cvzone.putTextRect(frame,f'{last_detected_person}',(x1,y1),1,1)
    #            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)    
    
   
    # with lock:
    #     ret, frame = cap.read()
    
    # count += 1
    # if count % 3 != 0:
    #     continue

    # results = model(frame)
    # a = results[0].boxes.data
    # px = pd.DataFrame(a).astype("float")

    # for index, row in px.iterrows():
    #     x1 = int(row[0])
    #     y1 = int(row[1])
    #     x2 = int(row[2])
    #     y2 = int(row[3])
        
    #     d = int(row[5])
    #     c = class_list[d]  # Class name (e.g., "person", "sitting", etc.)
    #     person_id = index  # Using index as a temporary identifier for a person

    #     # Improved text placement
    #     if "person" in c:
    #         cvzone.putTextRect(frame, f'{c}', (x1, y1 - 10), scale=1, thickness=2)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)    

    #     h = y2 - y1
    #     w = x2 - x1
    #     aspect_ratio = h / float(w)  # Aspect ratio check

    #     # If 'person' is detected
    #     if 'person' in c:
    #         # Update person's previous state
    #         if person_id not in person_states:
    #             person_states[person_id] = c  # Store initial state

    #         prev_state = person_states.get(person_id, "unknown")

    #         # Detect fall **ONLY IF the previous state was 'standing' or 'walking'**
    #         if prev_state in ["person", "walking"] and aspect_ratio < 0.4:  # Adjusted threshold
    #             cvzone.putTextRect(frame, f'{last_detected_person} FALL!', (x1, y1 - 30), scale=1.5, thickness=2, colorR=(0, 0, 255))
    #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box for fallen person

    #             # Save snapshot
    #             # save_fall_snapshot(frame, last_detected_person)

    #         else:
    #            cvzone.putTextRect(frame,f'{last_detected_person}',(x1,y1),1,1)
    #            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)   
    #         # Update the person's current state
    #         person_states[person_id] = c



    cv2.imshow("Fall Detection System with Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()

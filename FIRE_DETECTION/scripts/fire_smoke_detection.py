import cv2
import torch
from ultralytics import YOLO
import os
import time
from datetime import datetime

import requests

# Load the YOLOv8 model (ensure you have a trained model for fire & smoke detection)
model = YOLO("/Users/zoftcares/StudioProjects/AI PROJECTS/AITIQS_INCIDENTS_DETECTION/FIRE_DETECTION/best.pt")  # Replace with your trained model


RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
# cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/YOLOv8-Fire-and-Smoke-Detection/demo.mp4")
cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)

last_detection_time = {}  # Dictionary to store last detection timestamps per weapon

# Directory to save snapshots
SNAPSHOT_DIR = "fire_smoke_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

GRAPHQL_URL = "http://43.205.136.121:9001/graphql"


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
            "description": f'{service} detected!, need immediate attention.',
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


# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Break when video ends
    
    # Run YOLOv8 detection
    results = model(frame)

    # Process results and draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Define label (Assuming 0 = Fire, 1 = Smoke)
            label = "Fire" if cls == 0 else "Smoke"
            color = (0, 0, 255) if cls == 0 else (255, 255, 0)  # Red for fire, Blue for smoke
            
             # Cooldown logic
            current_time = time.time()
             # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if label not in last_detection_time or (current_time - last_detection_time[label] > 900):
                print(f"🟢 {label} DETECTION CONFIRMED. Saving snapshot...")
                # save_fall_snapshot(frame, label)  # Call the snapshot function
                last_detection_time[label] = current_time
            else:
                time_left = 900 - (current_time - last_detection_time[label])
                print(f"⏳ {label} detected again but ignored (Cooldown: {time_left:.2f} seconds left).")

           

    # Show frame
    cv2.imshow("Fire & Smoke Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

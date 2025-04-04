# import cv2
# from ultralytics import YOLO
# import os
# # model=YOLO('yolov10s.pt')

# model=YOLO('/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/scripts/runs/detect/cctv_fall_train/weights/best.pt')

# # RTSP Camera URL
# RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'

# # Set FFmpeg options
# os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp;timeout;5000000'
# cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# # cap=cv2.VideoCapture(0)
# while cap.isOpened():
#     ret,frame=cap.read()
#     if not ret:
#         break
#     results=model(frame)
    
#     for r in results:
#         for box in r.boxes:
#             x1,y1,x2,y2=map(int,box.xyxy[0])
#             confidence=box.conf[0].item()
#             class_id=int(box.cls[0].item())
#             label=model.names[class_id]

#             cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.putText(frame,f"{label} {confidence:.2f}", (x1,y1-10),
#                         cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            
#     cv2.imshow("yolo detection",frame)
#     if cv2.waitKey(1) & 0xFF==ord('q')  :
#         break
# cap.release()
# cv2.destroyAllWindows()
            

import cv2
from ultralytics import YOLO
import os
import requests
from datetime import datetime
import time
# model=YOLO('yolov10s.pt')

GRAPHQL_URL = "http://43.205.136.121:9001/graphql"

# model=YOLO('/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/scripts/runs/detect/hashir_cctv_fall(26-03-2025)/weights/best.pt')
model=YOLO('/Users/zoftcares/StudioProjects/AI PROJECTS/AITIQS_INCIDENTS_DETECTION/FALL_DETECTION/scripts/runs/detect/op_trained_model/weights/best.pt')

# RTSP Camera URL
RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'

# Set FFmpeg options
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp;'
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)




# Directory to save snapshots
# SNAPSHOT_DIR = "fall_snapshots"
# os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Initialize detection buffer and cooldown tracker
detection_buffer = []
last_detection_time = {}  # Dictionary to store last detection timestamps per weapon

GRAPHQL_URL = "http://43.205.136.121:9001/graphql"

# Directory to save snapshots
SNAPSHOT_DIR = "shemeer_snapshots"
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
            "description": f'person fall is detected! Need immediate attention',
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


# cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = model.names[class_id]

            # # Draw bounding box & label
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # # Cooldown logic
            # current_time = time.time()
            # if label not in last_detection_time or (current_time - last_detection_time[label] > 900):
            #     print(f"🟢 {label} DETECTION CONFIRMED. Saving snapshot...")
            #     save_fall_snapshot(frame, label)  # Call the snapshot function
            #     last_detection_time[label] = current_time
            # else:
            #     time_left = 900 - (current_time - last_detection_time[label])
            #     print(f"⏳ {label} detected again but ignored (Cooldown: {time_left:.2f} seconds left).")
            # Draw bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Cooldown logic (only for fall detection)
            if label.lower() == "fall":
                current_time = time.time()
                if label not in last_detection_time or (current_time - last_detection_time[label] > 900):
                    print(f"🟢 {label.upper()} DETECTION CONFIRMED. Saving snapshot...")
                    # save_fall_snapshot(frame, label)  # Call the snapshot function
                    last_detection_time[label] = current_time
                else:
                    time_left = 900 - (current_time - last_detection_time[label])
                    print(f"⏳ {label.upper()} detected again but ignored (Cooldown: {time_left:.2f} seconds left).")

    # Show the processed frame
    cv2.imshow("Live Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
            


# import cv2
# from ultralytics import YOLO
# import os
# import time
# from datetime import datetime
# import requests


# # Initialize detection buffer
# detection_buffer = []

# GRAPHQL_URL = "http://43.205.136.121:9001/graphql"

# # Directory to save snapshots
# SNAPSHOT_DIR = "weapon_snapshots"
# os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# # Function to capture and save snapshot
# def save_fall_snapshot(frame, weapon):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{SNAPSHOT_DIR}/{weapon}_weapon_{timestamp}.jpg"
#     cv2.imwrite(filename, frame)
#     print(f"📸 Snapshot saved: {filename}")
    
#     # Send GraphQL mutation
#     mutation = """
#         mutation CreateIncident($input: CreateIncidentInput!) {
#         createIncident(input: $input) {
#             incident {
#             id
#             }
#             message
#         }
#         }
#     """
    
#     # Define the variables for the mutation
#     variables = {
#         "input":{
#            "title": "Weapon detected!!!!!!",
#         "incidentType": 1,
#         "description": f'{weapon} is detected',
#         "urgency": None,
#         "priority": None,
#         "serviceId": "1",
#         "assignedUserId": None 
#         }
       
#     }
    
#     headers = {
#     "Content-Type": "application/json",
#     "x-tenant-id": "tenant4"
#     }

    
#     response = requests.post(GRAPHQL_URL, json={"query": mutation, "variables": variables},headers=headers)
    
#     print("RESPONSE BODY")
#     print(response.content)
    
#     if response.status_code == 200:
#         print("✅ GraphQL mutation sent successfully!")
#     else:
#         print("❌ Failed to send mutation:", response.text)

# def detect_objects_in_photo(image_path):
#     image_orig = cv2.imread(image_path)
    
#     yolo_model = YOLO('./runs/detect/Haar_Compressed/weights/best.pt')
    
#     results = yolo_model(image_orig)

#     for result in results:
#         classes = result.names
#         cls = result.boxes.cls
#         conf = result.boxes.conf
#         detections = result.boxes.xyxy

#         for pos, detection in enumerate(detections):
#             if conf[pos] >= 0.5:
#                 xmin, ymin, xmax, ymax = detection
#                 label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
#                 color = (0, int(cls[pos]), 255)
#                 cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
#                 cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

#     result_path = "./imgs/Test/teste.jpg"
#     cv2.imwrite(result_path, image_orig)
#     return result_path

# def detect_objects_in_video(video_path):
#     yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
#     video_capture = cv2.VideoCapture(video_path)
#     width = int(video_capture.get(3))
#     height = int(video_capture.get(4))
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     result_video_path = "detected_objects_video2.avi"
#     out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (width, height))

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         results = yolo_model(frame)

#         for result in results:
#             classes = result.names
#             cls = result.boxes.cls
#             conf = result.boxes.conf
#             detections = result.boxes.xyxy

#             for pos, detection in enumerate(detections):
#                 if conf[pos] >= 0.5:
#                     xmin, ymin, xmax, ymax = detection
#                     label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
#                     color = (0, int(cls[pos]), 255)
#                     cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
#                     cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

#         out.write(frame)
#     video_capture.release()
#     out.release()

#     return result_video_path

# def detect_objects_and_plot(path_orig):
#     image_orig = cv2.imread(path_orig)
    
#     yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    
#     results = yolo_model(image_orig)

#     for result in results:
#         classes = result.names
#         cls = result.boxes.cls
#         conf = result.boxes.conf
#         detections = result.boxes.xyxy

#         for pos, detection in enumerate(detections):
#             if conf[pos] >= 0.5:
#                 xmin, ymin, xmax, ymax = detection
#                 label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}" 
#                 color = (0, int(cls[pos]), 255)
#                 cv2.rectangle(image_orig, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
#                 cv2.putText(image_orig, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
#     cv2.imshow("Teste", image_orig)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def detect_objects_live():
    
#     global detection_buffer
    
#     # Load the YOLO model
#     yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')

#     # Start capturing video from webcam (0 is the default webcam)
#     # video_capture = cv2.VideoCapture(0)
    
#     RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'
#     os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
#     # cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/hafeefa2_fall.mp4")
#     # cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/fall2.mp4")
#     cap = cv2.VideoCapture(RTSP_URL,cv2.CAP_FFMPEG)       

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame.")
#             break

#         # Run YOLO inference
#         results = yolo_model(frame)

#         for result in results:
#             classes = result.names
#             cls = result.boxes.cls
#             conf = result.boxes.conf
#             detections = result.boxes.xyxy

#             for pos, detection in enumerate(detections):
#                 if conf[pos] >= 0.6:
#                     xmin, ymin, xmax, ymax = map(int, detection)
#                     label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
#                     color = (0, 255, 0)  # Green color for bounding boxes
#                     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
#                     cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    
#                      # Add detection to buffer
#                     detection_buffer.append((frame.copy(), classes[int(cls[pos])]))
                    
#                      # If buffer reaches 10 detections, save snapshot and clear buffer
#                     if len(detection_buffer) >= 100:
#                         snapshot_frame, detected_class = detection_buffer[0]
#                         # save_fall_snapshot(snapshot_frame, detected_class)
#                         detection_buffer.clear()
                    
#                     # save_fall_snapshot(frame,classes[int(cls[pos])])

#         # Show the processed frame
#         cv2.imshow("Live Object Detection", frame)

#         # Press 'q' to exit
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()

# # Run live detection
# detect_objects_live()
    


# # video_path = "Results/detected_objects_video.mp4"  # Update with your video path
# # result_video_path = detect_objects_in_video(video_path)
# # print(f"Processed video saved at: {result_video_path}")



# import cv2
# from ultralytics import YOLO
# import os
# import time
# from datetime import datetime
# import requests

# # Initialize detection buffer and cooldown tracker
# detection_buffer = []
# last_detection_time = {}  # Dictionary to store the last detection timestamp for each object

# GRAPHQL_URL = "http://43.205.136.121:9001/graphql"

# # Directory to save snapshots
# SNAPSHOT_DIR = "weapon_snapshots"
# os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# # Function to capture and save snapshot
# def save_fall_snapshot(frame, weapon):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{SNAPSHOT_DIR}/{weapon}_weapon_{timestamp}.jpg"
#     cv2.imwrite(filename, frame)
#     print(f"📸 Snapshot saved: {filename}")

#     # Send GraphQL mutation
#     mutation = """
#         mutation CreateIncident($input: CreateIncidentInput!) {
#         createIncident(input: $input) {
#             incident {
#             id
#             }
#             message
#         }
#         }
#     """
    
#     variables = {
#         "input": {
#             "title": f"{weapon} detected!!!!!!",
#             "incidentType": 1,
#             "description": f'{weapon} is detected',
#             "urgency": None,
#             "priority": None,
#             "serviceId": "1",
#             "assignedUserId": None
#         }
#     }
    
#     headers = {
#         "Content-Type": "application/json",
#         "x-tenant-id": "tenant4"
#     }

#     response = requests.post(GRAPHQL_URL, json={"query": mutation, "variables": variables}, headers=headers)

#     if response.status_code == 200:
#         print(f"✅ {weapon} alert sent successfully!")
#     else:
#         print(f"❌ Failed to send {weapon} alert:", response.text)

# # Function for live object detection with per-weapon cooldown
# def detect_objects_live():
#     global detection_buffer, last_detection_time

#     yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')

#     RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'
#     os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
#     cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame.")
#             break

#         results = yolo_model(frame)

#         for result in results:
#             classes = result.names
#             cls = result.boxes.cls
#             conf = result.boxes.conf
#             detections = result.boxes.xyxy

#             for pos, detection in enumerate(detections):
#                 if conf[pos] >= 0.6:
#                     xmin, ymin, xmax, ymax = map(int, detection)
#                     detected_class = classes[int(cls[pos])]  # Example: "gun", "knife"
#                     label = f"{detected_class} {conf[pos]:.2f}"
#                     color = (0, 255, 0)  # Green color for bounding boxes
#                     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
#                     cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

#                     # Check cooldown for this specific weapon
#                     current_time = time.time()
#                     if detected_class not in last_detection_time or (current_time - last_detection_time[detected_class] > 900):
#                         print(f"🚨 {detected_class} detected! Saving snapshot.")
#                         save_fall_snapshot(frame, detected_class)
#                         last_detection_time[detected_class] = current_time  # Update last detection time
#                     else:
#                         time_left = 900 - (current_time - last_detection_time[detected_class])
#                         print(f"⚠️ {detected_class} detected again but ignored (Cooldown: {time_left:.2f} seconds left).")

#         cv2.imshow("Live Object Detection", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Run live detection
# detect_objects_live()


import cv2
from ultralytics import YOLO
import os
import time
from datetime import datetime
import requests

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
def save_fall_snapshot(frame, weapon):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SNAPSHOT_DIR}/{weapon}_weapon_{timestamp}.jpg"
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
            "title": f"{weapon} detected!!!!!!",
            "incidentTypeId": "1",
            "description": f'{weapon} detected in live camera!, Need immediate attention..',
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
        print(f"✅ {weapon} alert sent successfully!")
    else:
        print(f"❌ Failed to send {weapon} alert:", response.text)

    

# Function to capture and save snapshot
# # Function to capture and save snapshot
# def save_fall_snapshot(frame, weapon):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{SNAPSHOT_DIR}/{weapon}_weapon_{timestamp}.jpg"
#     cv2.imwrite(filename, frame)
#     print(f"📸 Snapshot saved: {filename}")

#     # Send GraphQL mutation
#     mutation = """
#         mutation CreateIncident($input: CreateIncidentInput!) {
#         createIncident(input: $input) {
#             incident {
#             id
#             }
#             message
#         }
#         }
#     """
    
#     variables = {
#         "input": {
#             "title": f"{weapon} detected!!!!!!",
#             "incidentType": 1,
#             "description": f'{weapon} is detected',
#             "urgency": None,
#             "priority": None,
#             "serviceId": "1",
#             "assignedUserId": None
#         }
#     }
    
#     headers = {
#         "Content-Type": "application/json",
#         "x-tenant-id": "zoftcares"
#     }

#     response = requests.post(GRAPHQL_URL, json={"query": mutation, "variables": variables}, headers=headers)

#     if response.status_code == 200:
#         print(f"✅ {weapon} alert sent successfully!")
#     else:
#         print(f"❌ Failed to send {weapon} alert:", response.text)

# Live detection with proper bounding box display
def detect_objects_live():
    global detection_buffer, last_detection_time

    yolo_model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')

    RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        results = yolo_model(frame)

        for result in results:
            classes = result.names
            cls = result.boxes.cls.cpu().numpy()  # Ensure it's converted properly
            conf = result.boxes.conf.cpu().numpy()
            detections = result.boxes.xyxy.cpu().numpy()  # Ensure proper format

            for pos, detection in enumerate(detections):
                if conf[pos] >= 0.7:  # Confidence threshold
                    xmin, ymin, xmax, ymax = map(int, detection)  # Convert to integers
                    detected_class = classes[int(cls[pos])]  # Convert class index to label
                    
                    # Debugging print
                    print(f"🚨 DETECTED: {detected_class} at [{xmin}, {ymin}, {xmax}, {ymax}] (Confidence: {conf[pos]:.2f})")

                    # Draw bounding box & label
                    # color = (0, 255, 0)  # Green box
                    color = (0,0,255) # Red box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 5)  # Increased thickness
                    cv2.putText(frame, f"{detected_class} {conf[pos]:.2f}", 
                            (xmin, ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)  # Increased font size & bold

                    # Check cooldown
                    current_time = time.time()
                    if detected_class not in last_detection_time or (current_time - last_detection_time[detected_class] > 900):
                        print(f"🟢 {detected_class} DETECTION CONFIRMED. Saving snapshot...")
                        save_fall_snapshot(frame, detected_class)
                        last_detection_time[detected_class] = current_time
                    else:
                        time_left = 900 - (current_time - last_detection_time[detected_class])
                        print(f"⏳ {detected_class} detected again but ignored (Cooldown: {time_left:.2f} seconds left).")

        # Show the processed frame
        cv2.imshow("Live Object Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run live detection
detect_objects_live()

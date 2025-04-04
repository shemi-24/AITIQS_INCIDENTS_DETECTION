import cv2
from ultralytics import YOLO
import os

model=YOLO('/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/scripts/runs/detect/train_weapon_without_human/weights/best.pt')
# cap = cv2.VideoCapture(0)  # 0 for default webcam, change for external camera

RTSP_URL = 'rtsp://rahil:Rahil123@112.133.238.189:554/Streaming/channels/301'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
# cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/hafeefa2_fall.mp4")
# cap = cv2.VideoCapture("/Users/zoftcares/StudioProjects/AI PROJECTS/fall_detection_system/fall2.mp4")
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

while cap.isOpened():
    ret, frame = cap.read()  # Read frame from webcam
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Process results
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
             #  Correct way: Process each detected object one by one
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID
            label = model.names[class_id]  # Object label (e.g., "person", "car") # index kittan aan 
            
            print(label)    

            # # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # This part is used for positioning the label text above the bounding box.
             # Save snapshot
            # save_fall_snapshot(frame, last_detected_person)

    # Display the frame with detections
    cv2.imshow("YOLOv10 Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
# .item() converts the tensor value into a standard Python data type (float)

# Why use an integer?
#-------------------------
# Class IDs are always whole numbers (0, 1, 2, ...).
# model.names is a list or dictionary indexed by integers, so we must pass an integer to access the correct label.

# The first loop picks the first detected object → box.cls[0] gets its class.
# 🔹 The second loop picks the next object → box.cls[0] gets its class.
# 🔹 This repeats for all detected objects.

# So even though we always use [0], we get different classes because we are processing each object separately.
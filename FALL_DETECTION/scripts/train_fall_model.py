from ultralytics import YOLO

# model = YOLO("yolov8n.pt")  # Load YOLO model
model = YOLO("yolo11s.pt")  # Load YOLO model
pose_model = YOLO("yolo11s-pose.pt")

model.train(data="/Users/zoftcares/StudioProjects/fall_detection_system/model/yolo11_face.yaml", epochs=100, imgsz=320, batch=8)
model.train(data="/Users/zoftcares/StudioProjects/fall_detection_system/model/yolo11_fall.yaml", epochs=100, imgsz=320, batch=8)
# model.train(data="/Users/zoftcares/StudioProjects/fall_detection_system/model/yolo11_fall_mediapipe.yaml", epochs=100, imgsz=320, batch=8)

pose_model.train(data="/Users/zoftcares/StudioProjects/fall_detection_system/model/yolo11_pose.yaml",epochs=100, imgsz=320, batch=8)


# Save trained model
model.export(format="torchscript")
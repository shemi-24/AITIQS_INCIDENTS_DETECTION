from ultralytics import YOLO

# model = YOLO("yolov8n.pt")  # Load YOLO model
model = YOLO("yolo11s.pt")  # Load YOLO model

model.train(data="model/yolo11_weapon.yaml",epochs=100, imgsz=640, batch=16)


# Save trained model
model.export(format="torchscript")
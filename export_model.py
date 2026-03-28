from ultralytics import YOLO

# Зарежда pretrained модел
model = YOLO("yolov8n.pt")

# Експорт към ONNX
model.export(format="onnx")
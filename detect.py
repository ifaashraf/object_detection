from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model (auto-downloads on first run)
model = YOLO("yolov8n.pt")

# Run detection on an image
results = model("emi and me.jpeg", conf=0.5)  # replace with your image path

# Show result with bounding boxes
results[0].show()

# Or save it
results[0].save(filename="result.jpg")
from pathlib import Path
import sys

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO
from severity.severity import calculate_severity
from PIL import Image


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "road_damage_yolo11n_best.pt"
IMAGE_PATH = BASE_DIR / "test_images" / "new_road.jpeg"


# -----------------------------
# Load model
# -----------------------------

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("Image:", IMAGE_PATH)


# -----------------------------
# Run YOLO
# -----------------------------

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    verbose=False
)


# -----------------------------
# Extract detections
# -----------------------------

detections = []

result = results[0]

for box in result.boxes:

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    damage = model.names[class_id]

    detections.append({
        "damage": damage,
        "confidence": confidence,
        "bounding_box": [x1, y1, x2, y2]
    })


# -----------------------------
# Get image dimensions
# -----------------------------

with Image.open(IMAGE_PATH) as image:
    image_width, image_height = image.size


# -----------------------------
# Print detections
# -----------------------------

print("\nDETECTIONS")
print("==============================")

for i, detection in enumerate(detections, start=1):

    print(f"\nDetection {i}")
    print("Damage     :", detection["damage"])
    print("Confidence :", round(detection["confidence"], 3))
    print("Bounding Box:", detection["bounding_box"])


# -----------------------------
# Calculate severity
# -----------------------------

severity = calculate_severity(
    detections=detections,
    image_width=image_width,
    image_height=image_height
)


# -----------------------------
# Final result
# -----------------------------

print("\n==============================")
print("ROAD DAMAGE ASSESSMENT")
print("==============================")

print("Severity Score :", severity["score"])
print("Severity Level :", severity["level"])
print("Reason         :", severity["reason"])
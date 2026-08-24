from pathlib import Path
import sys
from PIL import Image
from ultralytics import YOLO

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from severity.severity import calculate_severity


# -----------------------------
# Paths
# -----------------------------

MODEL_PATH = BASE_DIR / "model" / "road_damage_yolo11n_best.pt"
IMAGE_DIR = BASE_DIR / "test_images"


# -----------------------------
# Load model
# -----------------------------

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("=" * 60)


# -----------------------------
# Find images
# -----------------------------

image_extensions = {".jpg", ".jpeg", ".png"}

images = [
    image for image in IMAGE_DIR.iterdir()
    if image.suffix.lower() in image_extensions
]

if not images:
    print("No images found in test_images.")
    sys.exit()


# -----------------------------
# Test every image
# -----------------------------

for image_path in images:

    print("\n")
    print("=" * 60)
    print("IMAGE:", image_path.name)
    print("=" * 60)

    results = model.predict(
        source=str(image_path),
        conf=0.40,
        verbose=False
    )

    result = results[0]

    detections = []

    # Extract YOLO detections
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

    # Get image dimensions
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    # Print detections
    if detections:

        print("\nDETECTIONS:")

        for i, detection in enumerate(detections, start=1):

            print(
                f"  {i}. "
                f"{detection['damage']} "
                f"(confidence: {detection['confidence']:.3f})"
            )

    else:

        print("\nDETECTIONS:")
        print("  No damage detected.")

    # Calculate severity
    severity = calculate_severity(
        detections=detections,
        image_width=image_width,
        image_height=image_height
    )

    # Print severity
    print("\nSEVERITY:")
    print("  Score :", severity["score"])
    print("  Level :", severity["level"])
    print("  Reason:", severity["reason"])


print("\n")
print("=" * 60)
print("MULTI-IMAGE TEST COMPLETE")
print("=" * 60)
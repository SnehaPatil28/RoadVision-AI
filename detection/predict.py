from pathlib import Path

from ultralytics import YOLO

from severity.severity import (
    calculate_severity,
    calculate_road_condition_score,
    get_road_condition,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model" / "road_damage_yolo11n_best.pt"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "predictions"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found at {MODEL_PATH}. "
        "Add the trained checkpoint under model/road_damage_yolo11n_best.pt."
    )

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully.")
print("\nModel classes:")
print(model.names)


# ============================================================
# GET TEST IMAGES
# ============================================================

image_files = sorted(
    list(TEST_IMAGES_DIR.glob("*.jpg"))
    + list(TEST_IMAGES_DIR.glob("*.jpeg"))
    + list(TEST_IMAGES_DIR.glob("*.png"))
)


if not image_files:
    print("\nNo test images found.")
    print("Please add images to:", TEST_IMAGES_DIR)
    raise SystemExit


print("\nTest images found:")

for image in image_files:
    print("-", image.name)


# ============================================================
# RUN PREDICTION
# ============================================================

print("\nRunning prediction...")

results = model.predict(
    source=[str(image) for image in image_files],
    conf=0.20,
    save=True,
    project=str(OUTPUT_DIR),
    name="prediction",
    exist_ok=True,
    verbose=True,
)


# ============================================================
# PROCESS EVERY IMAGE
# ============================================================

print("\n")
print("=" * 70)
print("ROAD DAMAGE BATCH REPORT")
print("=" * 70)

all_reports = []

for image_path, result in zip(image_files, results):
    print("\n")
    print("=" * 70)
    print(f"IMAGE: {image_path.name}")
    print("=" * 70)

    image_height, image_width = result.orig_shape
    print(f"\nImage dimensions: {image_width} x {image_height}")

    if result.boxes is None or len(result.boxes) == 0:
        road_score = 100.0
        road_condition = get_road_condition(road_score)

        print("\nNo damage detected.")
        print(f"Road Condition Score: {road_score:.2f} / 100")
        print(f"Road Condition: {road_condition}")

        all_reports.append({
            "image": image_path.name,
            "detections": 0,
            "score": road_score,
            "condition": road_condition,
        })
        continue

    print(f"\nTotal detections: {len(result.boxes)}")
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        bbox = [float(x) for x in box.xyxy[0].cpu().numpy()]
        damage_type = model.names[class_id]

        current_severity = calculate_severity(
            [{
                "damage": damage_type,
                "confidence": confidence,
                "bounding_box": bbox,
            }],
            image_width,
            image_height,
        )

        detection = {
            "damage_type": damage_type,
            "confidence": round(confidence, 3),
            "bbox": [round(x, 2) for x in bbox],
            "coverage": round(min(100.0, (max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])) / max(1, image_width * image_height) * 100.0), 2),
            "severity": current_severity["level"],
            "severity_score": current_severity["score"],
        }

        detections.append(detection)

    road_score = calculate_road_condition_score(detections)
    road_condition = get_road_condition(road_score)

    for i, detection in enumerate(detections, start=1):
        print(f"\nDetection {i}")
        print("-" * 35)
        print("Damage:", detection["damage_type"])
        print("Confidence:", detection["confidence"])
        print("Bounding Box:", detection["bbox"])
        print("Damage Coverage:", f'{detection["coverage"]}%')
        print("Severity:", detection["severity"])

    print("\n" + "-" * 50)
    print("Road Condition Score:", f"{road_score:.2f} / 100")
    print("Road Condition:", road_condition)

    all_reports.append({
        "image": image_path.name,
        "detections": len(detections),
        "score": round(road_score, 2),
        "condition": road_condition,
    })


print("\n")
print("=" * 70)
print("FINAL BATCH SUMMARY")
print("=" * 70)

for report in all_reports:
    print(f"\n{report['image']}")
    print(f"Detections: {report['detections']}")
    print(f"Score: {report['score']} / 100")
    print(f"Condition: {report['condition']}")

print("\n")
print("=" * 70)
print("Annotated images saved to:")
print(OUTPUT_DIR / "prediction")
print("=" * 70)

from pathlib import Path

from ultralytics import YOLO

from severity.severity import (
    calculate_severity,
    calculate_road_condition_score,
    get_road_condition
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

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

    print(
        "Please add images to:",
        TEST_IMAGES_DIR
    )

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

    # Confidence threshold
    conf=0.20,

    # Save annotated images
    save=True,

    project=str(OUTPUT_DIR),
    name="prediction",
    exist_ok=True,

    verbose=True
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

    print(
        f"IMAGE: {image_path.name}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    image_height, image_width = result.orig_shape

    print(
        f"\nImage dimensions: "
        f"{image_width} x {image_height}"
    )

    # --------------------------------------------------------
    # No detections
    # --------------------------------------------------------

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        road_score = 100.0
        road_condition = get_road_condition(
            road_score
        )

        print("\nNo damage detected.")

        print(
            f"Road Condition Score: "
            f"{road_score:.2f} / 100"
        )

        print(
            f"Road Condition: "
            f"{road_condition}"
        )

        all_reports.append({

            "image": image_path.name,

            "detections": 0,

            "score": road_score,

            "condition": road_condition

        })

        continue

    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    print(
        f"\nTotal detections: "
        f"{len(result.boxes)}"
    )

    detections = []

    for i, box in enumerate(
        result.boxes,
        start=1
    ):

        # ----------------------------------------------------
        # Class ID
        # ----------------------------------------------------

        class_id = int(
            box.cls[0].item()
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            box.conf[0].item()
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        bbox = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        bbox = [
            float(x)
            for x in bbox
        ]

        # ----------------------------------------------------
        # Damage type
        # ----------------------------------------------------

        damage_type = model.names[class_id]

        # ----------------------------------------------------
        # Calculate severity
        # ----------------------------------------------------

        severity, coverage = calculate_severity(

            damage_type,

            bbox,

            image_width,

            image_height,

            confidence

        )

        # ----------------------------------------------------
        # Create detection dictionary
        # ----------------------------------------------------

        detection = {

            "damage_type": damage_type,

            "confidence": round(
                confidence,
                3
            ),

            "bbox": [
                round(x, 2)
                for x in bbox
            ],

            "coverage": round(
                coverage,
                2
            ),

            "severity": severity

        }

        # ----------------------------------------------------
        # Store detection
        # ----------------------------------------------------

        detections.append(
            detection
        )

    # ========================================================
    # ROAD CONDITION SCORE
    # ========================================================

    # IMPORTANT:
    # Pass the complete detection dictionaries,
    # NOT just the severity strings.

    road_score = calculate_road_condition_score(
        detections
    )

    # ========================================================
    # ROAD CONDITION CATEGORY
    # ========================================================

    road_condition = get_road_condition(
        road_score
    )

    # ========================================================
    # PRINT DETECTIONS
    # ========================================================

    for i, detection in enumerate(
        detections,
        start=1
    ):

        print(
            f"\nDetection {i}"
        )

        print(
            "-" * 35
        )

        print(
            "Damage:",
            detection["damage_type"]
        )

        print(
            "Confidence:",
            detection["confidence"]
        )

        print(
            "Bounding Box:",
            detection["bbox"]
        )

        print(
            "Damage Coverage:",
            f'{detection["coverage"]}%'
        )

        print(
            "Severity:",
            detection["severity"]
        )

    # ========================================================
    # PRINT ROAD RESULT
    # ========================================================

    print(
        "\n" + "-" * 50
    )

    print(
        "Road Condition Score:",
        f"{road_score:.2f} / 100"
    )

    print(
        "Road Condition:",
        road_condition
    )

    # ========================================================
    # STORE REPORT
    # ========================================================

    all_reports.append({

        "image": image_path.name,

        "detections": len(detections),

        "score": round(
            road_score,
            2
        ),

        "condition": road_condition

    })


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL BATCH SUMMARY")
print("=" * 70)


for report in all_reports:

    print(
        f"\n{report['image']}"
    )

    print(
        f"Detections: "
        f"{report['detections']}"
    )

    print(
        f"Score: "
        f"{report['score']} / 100"
    )

    print(
        f"Condition: "
        f"{report['condition']}"
    )


# ============================================================
# OUTPUT LOCATION
# ============================================================

print("\n")
print("=" * 70)

print(
    "Annotated images saved to:"
)

print(
    OUTPUT_DIR / "prediction"
)

print("=" * 70)

from pathlib import Path
import sys

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from PIL import Image
from ultralytics import YOLO
from severity.severity import calculate_severity


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "road_damage_yolo11n_best.pt"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

# Collect low-confidence predictions as possible damage.
# Severity.py will decide which detections are reliable.
CONFIDENCE_THRESHOLD = 0.10

# Confidence considered reliable/confirmed.
SEVERITY_CONFIDENCE_THRESHOLD = 0.40


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = YOLO(str(MODEL_PATH))


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def analyze_image(image_path):
    """
    Run complete RoadVision analysis on one image.

    Returns:
        {
            "image": image_path,
            "detections": [...],
            "confirmed_detections": [...],
            "possible_detections": [...],
            "severity": {...}
        }
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # ----------------------------------------------
    # Run YOLO
    # ----------------------------------------------

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]

    # ----------------------------------------------
    # Extract detections
    # ----------------------------------------------

    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        damage = model.names[class_id]

        detections.append({
            "class_id": class_id,
            "damage": damage,
            "confidence": round(confidence, 3),
            "bounding_box": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ]
        })

    # ----------------------------------------------
    # Separate confirmed and possible detections
    # ----------------------------------------------

    confirmed_detections = [
        detection
        for detection in detections
        if detection["confidence"]
        >= SEVERITY_CONFIDENCE_THRESHOLD
    ]

    possible_detections = [
        detection
        for detection in detections
        if detection["confidence"]
        < SEVERITY_CONFIDENCE_THRESHOLD
    ]

    # ----------------------------------------------
    # Get image dimensions
    # ----------------------------------------------

    with Image.open(image_path) as image:

        image_width, image_height = image.size

    # ----------------------------------------------
    # Calculate severity
    # ----------------------------------------------

    severity = calculate_severity(
        detections=detections,
        image_width=image_width,
        image_height=image_height
    )

    # ----------------------------------------------
    # Final result
    # ----------------------------------------------

    return {
        "image": str(image_path),
        "image_width": image_width,
        "image_height": image_height,
        "detections": detections,
        "confirmed_detections": confirmed_detections,
        "possible_detections": possible_detections,
        "severity": severity
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_image = (
        BASE_DIR
        / "test_images"
        / "new_road.jpeg"
    )

    print("Running RoadVision pipeline...")
    print("Image:", test_image)

    result = analyze_image(test_image)

    print("\n==============================")
    print("ROADVISION RESULT")
    print("==============================")

    print("\nALL DETECTIONS:")

    for i, detection in enumerate(
        result["detections"],
        start=1
    ):

        print(
            f"{i}. "
            f"{detection['damage']} "
            f"({detection['confidence']})"
        )

    print("\nCONFIRMED DAMAGE:")

    for detection in result["confirmed_detections"]:

        print(
            f"- {detection['damage']} "
            f"({detection['confidence']})"
        )

    print("\nPOSSIBLE DAMAGE:")

    for detection in result["possible_detections"]:

        print(
            f"- {detection['damage']} "
            f"({detection['confidence']})"
        )

    print("\nSeverity:")
    print(
        "Score:",
        result["severity"]["score"]
    )

    print(
        "Level:",
        result["severity"]["level"]
    )

    print(
        "Reason:",
        result["severity"]["reason"]
    )
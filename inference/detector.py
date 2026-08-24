from pathlib import Path
from ultralytics import YOLO


# -----------------------------
# MODEL PATH
# -----------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "road_damage_yolo11n_best.pt"
)


# -----------------------------
# CLASS NAMES
# -----------------------------

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
}


# -----------------------------
# LOAD MODEL
# -----------------------------

model = YOLO(str(MODEL_PATH))


# -----------------------------
# DETECTION FUNCTION
# -----------------------------

def detect_damage(image_path, confidence=0.30):

    results = model.predict(
        source=str(image_path),
        conf=confidence,
        device="cpu",
        verbose=False
    )

    result = results[0]

    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence_score = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": confidence_score,
            "bbox": [x1, y1, x2, y2],
        })

    return detections


# -----------------------------
# BASIC TEST
# -----------------------------

if __name__ == "__main__":

    print("Model path:")
    print(MODEL_PATH)

    print("\nModel exists:", MODEL_PATH.exists())

    if MODEL_PATH.exists():
        print("✅ Model found!")
        print("Classes:", CLASS_NAMES)
    else:
        print("❌ Model not found!")
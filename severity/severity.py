# Severity weights for each damage type
DAMAGE_WEIGHTS = {
    "Longitudinal Crack": 2,
    "Transverse Crack": 2,
    "Alligator Crack": 3,
    "Pothole": 3,
}

# Minimum confidence considered reliable for severity calculation.
# Lowered from 0.40 → 0.25 so moderate-confidence crack detections
# (common in real images) are not silently discarded.
MIN_CONFIDENCE = 0.25


def calculate_severity(detections, image_width, image_height):
    """
    Calculate road damage severity from YOLO detections.

    Each detection should contain:
        - damage
        - confidence
        - bounding_box [x1, y1, x2, y2]
    """

    if not detections:
        return {
            "score": 0,
            "level": "No Damage",
            "reason": "No road damage detected."
        }

    image_area = image_width * image_height

    total_score = 0
    valid_detections = []

    for detection in detections:

        damage = detection["damage"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["bounding_box"]

        # Ignore low-confidence detections
        if confidence < MIN_CONFIDENCE:
            continue

        # Bounding-box area
        box_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Percentage of image occupied by detected damage
        area_ratio = box_area / image_area

        # Damage-type weight
        weight = DAMAGE_WEIGHTS.get(damage, 1)

        # Contribution of this detection.
        # Confidence is dampened with sqrt() so lower-confidence detections
        # still contribute meaningfully without dominating the score.
        import math
        detection_score = weight * math.sqrt(confidence) * (1 + area_ratio)

        total_score += detection_score
        valid_detections.append(detection)

    # No reliable detections
    if not valid_detections:
        return {
            "score": 0,
            "level": "No Damage",
            "reason": "No reliable road damage detected."
        }

    # Limit score to 10
    severity_score = min(total_score, 10)

    # Convert score into severity category.
    # Bands widened slightly so a small confidence shift doesn't flip the level.
    if severity_score < 1.5:
        level = "Low"
    elif severity_score < 4.5:
        level = "Moderate"
    else:
        level = "Severe"

    # Count damage types
    damage_counts = {}

    for detection in valid_detections:
        damage = detection["damage"]
        damage_counts[damage] = damage_counts.get(damage, 0) + 1

    damage_summary = ", ".join(
        f"{count} {damage}"
        for damage, count in damage_counts.items()
    )

    return {
        "score": round(severity_score, 2),
        "level": level,
        "reason": f"{damage_summary} detected.",
        "damage_count": len(valid_detections)
    }


def calculate_road_condition_score(detections):
    """Return a percentage score from 0-100 based on detected road damage."""

    if not detections:
        return 100.0

    damage_coverage = 0.0

    for detection in detections:
        if isinstance(detection, dict):
            if "coverage" in detection:
                damage_coverage += float(detection["coverage"])
            elif "severity" in detection:
                severity = detection["severity"]
                if isinstance(severity, dict):
                    damage_coverage += float(severity.get("score", 0.0))
                else:
                    damage_coverage += float(severity)
            elif "confidence" in detection:
                damage_coverage += float(detection["confidence"]) * 100.0

    return max(0.0, 100.0 - min(100.0, damage_coverage))


def get_road_condition(score):
    """Convert a 0-100 condition score into a label."""

    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 55:
        return "Fair"
    if score >= 35:
        return "Poor"
    return "Critical"

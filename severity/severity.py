def calculate_severity(
    bbox,
    image_width,
    image_height,
    confidence,
    damage_type
):
    """
    Calculate damage severity based on
    bounding-box coverage and confidence.
    """

    x1, y1, x2, y2 = bbox

    # Calculate bounding-box dimensions
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)

    # Calculate damage area
    damage_area = box_width * box_height

    # Calculate total image area
    image_area = image_width * image_height

    # Calculate percentage of image covered by damage
    coverage = (damage_area / image_area) * 100

    # Determine severity
    if coverage < 1:
        severity = "LOW"

    elif coverage < 5:
        severity = "MEDIUM"

    else:
        severity = "HIGH"

    return {
        "damage_type": damage_type,
        "confidence": round(confidence, 3),
        "coverage_percent": round(coverage, 3),
        "severity": severity
    }


def detect_and_classify(
    image_path,
    model,
    confidence_threshold=0.25
):
    """
    Run YOLO detection and classify
    each detected damage by severity.
    """

    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        verbose=False
    )

    result = results[0]

    # Get original image dimensions
    image_height, image_width = result.orig_shape

    detections = []

    # No detections
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    # Process every detection
    for box in result.boxes:

        # Get class ID
        class_id = int(
            box.cls[0].item()
        )

        # Get damage class name
        damage_type = model.names[class_id]

        # Get confidence
        confidence = float(
            box.conf[0].item()
        )

        # Get bounding box
        bbox = box.xyxy[0].cpu().numpy()

        # Calculate severity
        severity_result = calculate_severity(
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
            confidence=confidence,
            damage_type=damage_type
        )

        # Add bounding box to result
        severity_result["bbox"] = [
            round(float(x), 2)
            for x in bbox
        ]

        detections.append(severity_result)

    return detections


def calculate_road_condition_score(detections):
    """
    Calculate overall road condition score
    from 0 to 100.

    Higher score = better road condition.
    """

    if not detections:
        return 100

    severity_points = {
        "LOW": 10,
        "MEDIUM": 25,
        "HIGH": 50
    }

    total_penalty = 0

    for detection in detections:

        severity = detection["severity"]
        confidence = detection["confidence"]

        penalty = (
            severity_points[severity]
            * confidence
        )

        total_penalty += penalty

    # Maximum penalty is 100
    total_penalty = min(
        total_penalty,
        100
    )

    score = 100 - total_penalty

    return round(score, 2)


def get_road_condition(score):
    """
    Convert numerical road score
    into a condition label.
    """

    if score >= 80:
        return "GOOD"

    elif score >= 60:
        return "MODERATE"

    elif score >= 40:
        return "POOR"

    else:
        return "CRITICAL"

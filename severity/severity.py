def calculate_severity(
    damage_type,
    bbox,
    image_width,
    image_height,
    confidence
):
    """
    Calculate damage severity using:
        - Damage type
        - Bounding-box coverage
        - Model confidence

    Returns:
        severity: LOW / MEDIUM / HIGH
        coverage: percentage of image covered by damage
    """

    x1, y1, x2, y2 = bbox

    # ========================================================
    # BOUNDING BOX DIMENSIONS
    # ========================================================

    box_width = max(
        0,
        x2 - x1
    )

    box_height = max(
        0,
        y2 - y1
    )

    # ========================================================
    # AREA CALCULATION
    # ========================================================

    box_area = (
        box_width * box_height
    )

    image_area = (
        image_width * image_height
    )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if image_area <= 0:

        return "LOW", 0.0

    # ========================================================
    # DAMAGE COVERAGE
    # ========================================================

    coverage = (
        box_area / image_area
    ) * 100

    # ========================================================
    # DAMAGE-SPECIFIC THRESHOLDS
    # ========================================================

    thresholds = {

        "Longitudinal Crack": {
            "medium": 3,
            "high": 8
        },

        "Transverse Crack": {
            "medium": 3,
            "high": 8
        },

        "Alligator Crack": {
            "medium": 5,
            "high": 15
        },

        "Pothole": {
            "medium": 3,
            "high": 10
        }

    }

    # Get thresholds for detected damage
    damage_thresholds = thresholds.get(
        damage_type,
        {
            "medium": 5,
            "high": 15
        }
    )

    medium_threshold = damage_thresholds["medium"]
    high_threshold = damage_thresholds["high"]

    # ========================================================
    # INITIAL SEVERITY BASED ON COVERAGE
    # ========================================================

    if coverage < medium_threshold:

        severity = "LOW"

    elif coverage < high_threshold:

        severity = "MEDIUM"

    else:

        severity = "HIGH"

    # ========================================================
    # CONFIDENCE ADJUSTMENT
    # ========================================================

    # Very low confidence should not increase severity.
    #
    # We only reduce severity when confidence is weak.
    #
    # This prevents an uncertain detection from being
    # classified as HIGH severity.

    if confidence < 0.40:

        if severity == "HIGH":

            severity = "MEDIUM"

        elif severity == "MEDIUM":

            severity = "LOW"

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return severity, coverage


# ============================================================
# ROAD CONDITION SCORE
# ============================================================

def calculate_road_condition_score(
    detections
):
    """
    Calculate overall road condition score.

    Score:
        100 = Excellent
        0   = Very Poor

    The score considers:
        - Severity
        - Detection confidence
        - Damage coverage
        - Number of detected damages

    detections must be a list of dictionaries.
    """

    # ========================================================
    # NO DAMAGE
    # ========================================================

    if not detections:

        return 100.0

    # ========================================================
    # SEVERITY PENALTIES
    # ========================================================

    severity_penalty = {

        "LOW": 5,

        "MEDIUM": 15,

        "HIGH": 30

    }

    total_penalty = 0.0

    # ========================================================
    # PROCESS EACH DETECTION
    # ========================================================

    for detection in detections:

        severity = detection["severity"]

        coverage = detection["coverage"]

        confidence = detection["confidence"]

        # ----------------------------------------------------
        # Base severity penalty
        # ----------------------------------------------------

        base_penalty = severity_penalty.get(
            severity,
            0
        )

        # ----------------------------------------------------
        # Confidence-weighted penalty
        # ----------------------------------------------------

        confidence_penalty = (
            base_penalty * confidence
        )

        # ----------------------------------------------------
        # Coverage penalty
        # ----------------------------------------------------

        coverage_penalty = (
            coverage * 0.5
        )

        # ----------------------------------------------------
        # Total penalty
        # ----------------------------------------------------

        penalty = (
            confidence_penalty
            + coverage_penalty
        )

        total_penalty += penalty

    # ========================================================
    # MULTIPLE DAMAGE PENALTY
    # ========================================================

    number_of_detections = len(
        detections
    )

    # Additional penalty for multiple damages.
    #
    # First detection has no additional penalty.
    # Every additional detection adds 2 points.

    if number_of_detections > 1:

        multiple_damage_penalty = (
            number_of_detections - 1
        ) * 2

        total_penalty += (
            multiple_damage_penalty
        )

    # ========================================================
    # CALCULATE FINAL SCORE
    # ========================================================

    score = (
        100 - total_penalty
    )

    # ========================================================
    # KEEP SCORE BETWEEN 0 AND 100
    # ========================================================

    score = max(
        0,
        min(
            100,
            score
        )
    )

    return round(
        score,
        2
    )


# ============================================================
# ROAD CONDITION CATEGORY
# ============================================================

def get_road_condition(
    score
):
    """
    Convert numerical score into a readable
    road-condition category.
    """

    if score >= 80:

        return "Excellent"

    elif score >= 60:

        return "Good"

    elif score >= 40:

        return "Moderate"

    elif score >= 20:

        return "Poor"

    else:

        return "Very Poor"

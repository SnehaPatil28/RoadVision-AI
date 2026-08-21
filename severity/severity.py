def calculate_severity(
    bbox,
    image_width,
    image_height
):
    """
    Calculate severity based on the percentage
    of image area covered by the detected damage.

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

    if image_area == 0:

        return "LOW", 0.0


    # ========================================================
    # DAMAGE COVERAGE
    # ========================================================

    coverage = (
        box_area / image_area
    ) * 100


    # ========================================================
    # SEVERITY CLASSIFICATION
    # ========================================================

    if coverage < 5:

        severity = "LOW"

    elif coverage < 15:

        severity = "MEDIUM"

    else:

        severity = "HIGH"


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


        # ----------------------------------------------------
        # Get detection information
        # ----------------------------------------------------

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
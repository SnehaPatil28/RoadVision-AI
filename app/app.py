import streamlit as st
import tempfile
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw

import sys

# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from inference.pipeline import analyze_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RoadVision",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.html("""
<style>

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}


/* HEADER */

.hero-title {
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.hero-subtitle {
    color: #64748b;
    font-size: 17px;
    margin-top: 2px;
    margin-bottom: 18px;
}


/* SECTION */

.section-title {
    font-size: 22px;
    font-weight: 750;
    margin-bottom: 12px;
}


/* RESULT CARD */

.result-card {
    padding: 22px;
    border-radius: 18px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-bottom: 14px;
}

.result-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 650;
    letter-spacing: 0.7px;
}

.result-value {
    font-size: 34px;
    font-weight: 800;
    margin-top: 5px;
}

.result-score {
    font-size: 28px;
    font-weight: 750;
    margin-top: 3px;
}


/* DAMAGE CARD */

.damage-card {
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    background: white;
    margin-bottom: 9px;
}

.damage-title {
    font-size: 17px;
    font-weight: 700;
}

.damage-info {
    color: #64748b;
    font-size: 14px;
    margin-top: 4px;
}


/* POSSIBLE DAMAGE */

.possible-card {
    padding: 14px 18px;
    border-radius: 14px;
    border: 1px dashed #f59e0b;
    background: #fffbeb;
    margin-bottom: 9px;
}

.possible-title {
    font-size: 17px;
    font-weight: 700;
}

.possible-info {
    color: #92400e;
    font-size: 14px;
    margin-top: 4px;
}


/* ASSESSMENT */

.assessment-card {
    padding: 20px;
    border-radius: 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.assessment-title {
    font-size: 18px;
    font-weight: 750;
}

.assessment-text {
    color: #64748b;
    margin-top: 7px;
}


/* BUTTON */

.stButton > button {
    border-radius: 10px;
    height: 45px;
    font-weight: 650;
}


/* IMAGE */

img {
    border-radius: 14px;
}


/* EXPANDER */

[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

</style>
""")


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "temp_path" not in st.session_state:
    st.session_state.temp_path = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(image_path, detections):

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    for detection in detections:

        x1, y1, x2, y2 = detection[
            "bounding_box"
        ]

        damage = detection[
            "damage"
        ]

        confidence = detection[
            "confidence"
        ]

        # Differentiate possible damage
        # using a dashed-style approximation
        is_possible = confidence < 0.40

        box_width = 3 if is_possible else 4

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="orange" if is_possible else "red",
            width=box_width
        )

        label_prefix = (
            "Possible "
            if is_possible
            else ""
        )

        label = (
            f"{label_prefix}"
            f"{damage} "
            f"{confidence:.2f}"
        )

        bbox = draw.textbbox(
            (x1, y1),
            label
        )

        draw.rectangle(
            bbox,
            fill="orange"
            if is_possible
            else "red"
        )

        draw.text(
            (x1, y1),
            label,
            fill="white"
        )

    return image


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">🚧 RoadVision</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'AI-Powered Road Damage Detection & Severity Assessment'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# SAVE IMAGE
# ============================================================

if uploaded_file:

    if (
        st.session_state.file_name
        != uploaded_file.name
    ):

        suffix = Path(
            uploaded_file.name
        ).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            st.session_state.temp_path = (
                temp_file.name
            )

        st.session_state.file_name = (
            uploaded_file.name
        )

        st.session_state.result = None


# ============================================================
# ANALYZE BUTTON
# ============================================================

if uploaded_file:

    control_col, spacer = st.columns(
        [1, 4]
    )

    with control_col:

        analyze_clicked = st.button(
            "🔍 Analyze Road",
            type="primary",
            width="stretch"
        )

    if analyze_clicked:

        with st.spinner(
            "Analyzing road damage..."
        ):

            st.session_state.result = (
                analyze_image(
                    st.session_state.temp_path
                )
            )

else:

    st.info(
        "Upload a road image to begin analysis."
    )

    st.stop()


# ============================================================
# BEFORE ANALYSIS
# ============================================================

if st.session_state.result is None:

    st.markdown(
        '<div class="section-title">'
        'Road Image'
        '</div>',
        unsafe_allow_html=True
    )

    st.image(
        uploaded_file,
        width="stretch"
    )

    st.info(
        "Click **Analyze Road** to start detection."
    )

    st.stop()


# ============================================================
# GET RESULTS
# ============================================================

result = st.session_state.result

detections = result[
    "detections"
]

confirmed_detections = result[
    "confirmed_detections"
]

possible_detections = result[
    "possible_detections"
]

severity = result[
    "severity"
]

score = severity[
    "score"
]

level = severity[
    "level"
]

reason = severity[
    "reason"
]

temp_path = (
    st.session_state.temp_path
)


# ============================================================
# SEVERITY ICON
# ============================================================

if level == "Severe":

    icon = "🔴"

elif level == "Moderate":

    icon = "🟠"

elif level == "Low":

    icon = "🟢"

else:

    icon = "⚪"


# ============================================================
# ROAD ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Road Analysis'
    '</div>',
    unsafe_allow_html=True
)

image_col, result_col = st.columns(
    [1.45, 1]
)


# ============================================================
# ORIGINAL IMAGE
# ============================================================

with image_col:

    st.markdown(
        "**Road Image**"
    )

    original_image = Image.open(
        temp_path
    )

    st.image(
        original_image,
        width="stretch"
    )


# ============================================================
# RESULT PANEL
# ============================================================

with result_col:

    st.html(f"""
    <div class="result-card">

        <div class="result-label">
            ROAD CONDITION
        </div>

        <div class="result-value">
            {icon} {level}
        </div>

        <div class="result-label">
            SEVERITY SCORE
        </div>

        <div class="result-score">
            {score} / 10
        </div>

    </div>
    """)


    # --------------------------------------------------------
    # CONFIRMED DAMAGE
    # --------------------------------------------------------

    st.markdown(
        "**Confirmed Damage**"
    )


    if confirmed_detections:

        damage_counts = Counter(
            d["damage"]
            for d in confirmed_detections
        )


        for damage, count in (
            damage_counts.items()
        ):

            confidences = [
                d["confidence"]
                for d in confirmed_detections
                if d["damage"] == damage
            ]

            highest_confidence = max(
                confidences
            )

            st.html(f"""
            <div class="damage-card">

                <div class="damage-title">
                    🔍 {damage}
                </div>

                <div class="damage-info">
                    {count} detected
                    &nbsp; • &nbsp;
                    Confidence:
                    {highest_confidence * 100:.1f}%
                </div>

            </div>
            """)

    else:

        st.info(
            "No confirmed damage detected."
        )


    # --------------------------------------------------------
    # POSSIBLE DAMAGE
    # --------------------------------------------------------

    if possible_detections:

        st.markdown(
            "**Possible Damage**"
        )


        for detection in possible_detections:

            damage = detection[
                "damage"
            ]

            confidence = detection[
                "confidence"
            ]

            st.html(f"""
            <div class="possible-card">

                <div class="possible-title">
                    ⚠️ Possible {damage}
                </div>

                <div class="possible-info">
                    Confidence:
                    {confidence * 100:.1f}%
                    &nbsp; • &nbsp;
                    Not included in severity
                </div>

            </div>
            """)


    # --------------------------------------------------------
    # ASSESSMENT
    # --------------------------------------------------------

    st.html(f"""
    <div class="assessment-card">

        <div class="assessment-title">
            {icon} Assessment
        </div>

        <div class="assessment-text">
            {reason}
        </div>

    </div>
    """)


# ============================================================
# DETECTION RESULT
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'Detection Result'
    '</div>',
    unsafe_allow_html=True
)

annotated_image = draw_detections(
    temp_path,
    detections
)

st.image(
    annotated_image,
    caption="AI Detected Road Damage",
    width="stretch"
)


# ============================================================
# DETAILED DETECTIONS
# ============================================================

if detections:

    st.divider()

    with st.expander(
        "🔎 View detailed detections"
    ):

        # ----------------------------------------------
        # CONFIRMED
        # ----------------------------------------------

        if confirmed_detections:

            st.markdown(
                "### Confirmed Damage"
            )

            for i, detection in enumerate(
                confirmed_detections,
                start=1
            ):

                damage = detection[
                    "damage"
                ]

                confidence = detection[
                    "confidence"
                ]

                st.write(
                    f"**{i}. {damage}**"
                )

                st.progress(
                    confidence,
                    text=(
                        f"{confidence * 100:.1f}% confidence"
                    )
                )


        # ----------------------------------------------
        # POSSIBLE
        # ----------------------------------------------

        if possible_detections:

            st.markdown(
                "### Possible Damage"
            )

            for detection in possible_detections:

                damage = detection[
                    "damage"
                ]

                confidence = detection[
                    "confidence"
                ]

                st.write(
                    f"⚠️ **Possible {damage}**"
                )

                st.progress(
                    confidence,
                    text=(
                        f"{confidence * 100:.1f}% confidence "
                        "(not used for severity)"
                    )
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RoadVision • AI-powered road damage analysis"
)
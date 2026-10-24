import streamlit as st
import tempfile
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw

import sys

# ============================================================
# PROJECT PATH  (unchanged)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from inference.pipeline import analyze_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RoadVision AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# GLOBAL CSS  — light background, navy accents, clean type
# ============================================================

st.markdown("""
<style>

/* --- Google Font --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* --- Page background --- */
.stApp {
    background-color: #f5f7fa;
}

/* --- Sidebar --- */
[data-testid="stSidebar"] {
    background-color: #1a2744;
    border-right: none;
}

[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

[data-testid="stSidebar"] .sidebar-brand {
    padding: 1.5rem 1.2rem 0.5rem 1.2rem;
    border-bottom: 1px solid #2d3f6b;
    margin-bottom: 1rem;
}

[data-testid="stSidebar"] .sidebar-brand h2 {
    color: #ffffff !important;
    font-size: 18px;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.3px;
}

[data-testid="stSidebar"] .sidebar-brand p {
    color: #94a3b8 !important;
    font-size: 12px;
    margin: 3px 0 0 0;
}

/* --- Main container --- */
.block-container {
    max-width: 1200px;
    padding: 2rem 2.5rem;
}

/* --- Page header --- */
.page-header {
    border-bottom: 2px solid #1a2744;
    padding-bottom: 12px;
    margin-bottom: 24px;
}

.page-header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #1a2744;
    margin: 0 0 2px 0;
}

.page-header p {
    font-size: 14px;
    color: #64748b;
    margin: 0;
}

/* --- Section heading --- */
.section-heading {
    font-size: 15px;
    font-weight: 600;
    color: #1a2744;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #e2e8f0;
}

/* --- Cards --- */
.info-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 12px;
}

.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #1a2744;
    border-radius: 8px;
    padding: 16px 18px;
    text-align: center;
}

.stat-card .stat-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}

.stat-card .stat-value {
    font-size: 28px;
    font-weight: 700;
    color: #1a2744;
    line-height: 1;
}

.stat-card .stat-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
}

/* Severity level colours */
.level-severe   { color: #b91c1c; }
.level-moderate { color: #b45309; }
.level-low      { color: #15803d; }
.level-none     { color: #475569; }

/* --- Damage row --- */
.damage-row {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #1a2744;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.damage-row.possible {
    border-left-color: #d97706;
    background: #fffdf7;
}

.damage-row .dr-name {
    font-size: 14px;
    font-weight: 600;
    color: #1e293b;
}

.damage-row .dr-meta {
    font-size: 13px;
    color: #64748b;
    margin-top: 2px;
}

.damage-row .dr-badge {
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    background: #eff6ff;
    color: #1d4ed8;
    white-space: nowrap;
}

.damage-row.possible .dr-badge {
    background: #fef3c7;
    color: #92400e;
}

/* --- Assessment box --- */
.assessment-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 18px;
}

.assessment-box .ab-title {
    font-size: 13px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}

.assessment-box .ab-text {
    font-size: 14px;
    color: #1e293b;
    line-height: 1.5;
}

/* --- Legend chip --- */
.legend {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: #475569;
    margin-right: 18px;
}

.legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    display: inline-block;
}

/* --- About table --- */
.about-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.about-table th {
    background: #f1f5f9;
    color: #374151;
    font-weight: 600;
    text-align: left;
    padding: 10px 14px;
    border-bottom: 2px solid #e2e8f0;
}
.about-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #f1f5f9;
    color: #374151;
    vertical-align: top;
}
.about-table tr:last-child td { border-bottom: none; }

/* --- Upload zone --- */
[data-testid="stFileUploader"] {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}

/* --- Buttons --- */
.stButton > button {
    border-radius: 6px;
    font-weight: 600;
    font-size: 14px;
    height: 42px;
}

/* --- Divider --- */
hr { border-color: #e2e8f0; }

/* Streamlit image caption */
.stImage figcaption {
    font-size: 12px;
    color: #94a3b8;
    text-align: center;
    margin-top: 4px;
}

img { border-radius: 6px; }

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE  (unchanged)
# ============================================================

if "result"    not in st.session_state: st.session_state.result    = None
if "temp_path" not in st.session_state: st.session_state.temp_path = None
if "file_name" not in st.session_state: st.session_state.file_name = None
if "page"      not in st.session_state: st.session_state.page      = "Dashboard"


# ============================================================
# DRAW DETECTIONS  (unchanged logic)
# ============================================================

def draw_detections(image_path, detections):

    image = Image.open(image_path).convert("RGB")
    draw  = ImageDraw.Draw(image)

    for detection in detections:

        x1, y1, x2, y2 = detection["bounding_box"]
        damage          = detection["damage"]
        confidence      = detection["confidence"]

        is_possible = confidence < 0.25
        box_width   = 3 if is_possible else 4

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="orange" if is_possible else "#c0392b",
            width=box_width
        )

        label_prefix = "Possible " if is_possible else ""
        label        = f"{label_prefix}{damage} {confidence:.2f}"

        bbox = draw.textbbox((x1, y1), label)
        draw.rectangle(bbox, fill="orange" if is_possible else "#c0392b")
        draw.text((x1, y1), label, fill="white")

    return image


# ============================================================
# SIDEBAR  — branding + navigation
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <h2>RoadVision AI</h2>
        <p>Road Damage Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Dashboard", "Road Inspection", "Results", "About Project"],
        index=["Dashboard", "Road Inspection", "Results", "About Project"].index(
            st.session_state.page
        ),
        label_visibility="collapsed"
    )

    st.session_state.page = page

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:11px; font-weight:600; color:#64748b; "
        "text-transform:uppercase; letter-spacing:0.7px; margin-bottom:5px;'>Status</div>",
        unsafe_allow_html=True
    )

    if st.session_state.result:
        st.markdown(
            "<small style='color:#6ee7b7;'>&#9679; Analysis complete</small>",
            unsafe_allow_html=True
        )
    elif st.session_state.temp_path:
        st.markdown(
            "<small style='color:#fcd34d;'>&#9679; Image loaded</small>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<small style='color:#94a3b8;'>&#9679; No image loaded</small>",
            unsafe_allow_html=True
        )


# ============================================================
# PAGE: DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown("""
    <div class="page-header">
        <h1>Dashboard</h1>
        <p>Overview of the RoadVision AI system</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Model</div>
            <div class="stat-value" style="font-size:18px; padding-top:4px;">YOLO11n</div>
            <div class="stat-sub">Object Detection</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Damage Classes</div>
            <div class="stat-value">4</div>
            <div class="stat-sub">Crack types + Pothole</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Severity Scale</div>
            <div class="stat-value">0 – 10</div>
            <div class="stat-sub">Area-weighted score</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        model_status = "Ready" if (BASE_DIR / "model" / "road_damage_yolo11n_best.pt").exists() else "Missing"
        colour       = "#15803d" if model_status == "Ready" else "#b91c1c"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Model Status</div>
            <div class="stat-value" style="color:{colour}; font-size:20px; padding-top:4px;">{model_status}</div>
            <div class="stat-sub">.pt weights file</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Detectable Damage Types</div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("""
        <div class="info-card">
            <div style="font-size:14px; font-weight:600; color:#1a2744; margin-bottom:6px;">Pothole</div>
            <div style="font-size:13px; color:#64748b;">Depressions or holes in the road surface caused by water infiltration and traffic wear. High severity weight (3).</div>
        </div>
        <div class="info-card">
            <div style="font-size:14px; font-weight:600; color:#1a2744; margin-bottom:6px;">Alligator Crack</div>
            <div style="font-size:13px; color:#64748b;">Interconnected crack pattern resembling alligator skin; indicates deep structural fatigue. High severity weight (3).</div>
        </div>""", unsafe_allow_html=True)

    with d2:
        st.markdown("""
        <div class="info-card">
            <div style="font-size:14px; font-weight:600; color:#1a2744; margin-bottom:6px;">Longitudinal Crack</div>
            <div style="font-size:13px; color:#64748b;">Cracks running parallel to the road centre line; typically caused by poor lane joint construction. Severity weight (2).</div>
        </div>
        <div class="info-card">
            <div style="font-size:14px; font-weight:600; color:#1a2744; margin-bottom:6px;">Transverse Crack</div>
            <div style="font-size:13px; color:#64748b;">Cracks perpendicular to the road centre line; caused by thermal expansion and contraction cycles. Severity weight (2).</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Severity Assessment Scale</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="info-card" style="border-top:3px solid #15803d;">
            <div style="font-size:14px; font-weight:700; color:#15803d;">Low  (0 – 1.5)</div>
            <div style="font-size:13px; color:#64748b; margin-top:6px;">Minor surface damage. Monitoring recommended; no urgent repair needed.</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="info-card" style="border-top:3px solid #b45309;">
            <div style="font-size:14px; font-weight:700; color:#b45309;">Moderate  (1.5 – 4.5)</div>
            <div style="font-size:13px; color:#64748b; margin-top:6px;">Notable road damage present. Schedule maintenance within 3 – 6 months.</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="info-card" style="border-top:3px solid #b91c1c;">
            <div style="font-size:14px; font-weight:700; color:#b91c1c;">Severe  (4.5 – 10)</div>
            <div style="font-size:13px; color:#64748b; margin-top:6px;">Significant structural damage. Immediate repair required for road safety.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style="text-align:center; font-size:13px; color:#94a3b8;">
        Navigate to <strong>Road Inspection</strong> to upload an image and run analysis.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE: ROAD INSPECTION
# ============================================================

elif page == "Road Inspection":

    st.markdown("""
    <div class="page-header">
        <h1>Road Inspection</h1>
        <p>Upload a road image to run YOLO damage detection</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select a road image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.markdown(
        "<div style='font-size:12px; color:#94a3b8; margin-top:4px;'>"
        "Accepted formats: JPG, JPEG, PNG. The model runs on CPU."
        "</div>",
        unsafe_allow_html=True
    )

    if uploaded_file:

        if st.session_state.file_name != uploaded_file.name:

            suffix = Path(uploaded_file.name).suffix

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                st.session_state.temp_path = tmp.name

            st.session_state.file_name = uploaded_file.name
            st.session_state.result    = None

        st.markdown('<div class="section-heading">Image Preview</div>', unsafe_allow_html=True)
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        btn_col, _ = st.columns([1, 3])
        with btn_col:
            run_analysis = st.button(
                "Run Detection",
                type="primary",
                use_container_width=True
            )

        if run_analysis:
            with st.spinner("Running YOLO detection and severity assessment..."):
                st.session_state.result = analyze_image(st.session_state.temp_path)

            # Automatically navigate to the Results page
            st.session_state.page = "Results"
            st.rerun()

    else:
        st.markdown("""
        <div class="info-card" style="text-align:center; padding:32px; color:#64748b;">
            No image uploaded yet.<br>
            <span style="font-size:13px;">Use the file picker above to select a road image.</span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PAGE: RESULTS
# ============================================================

elif page == "Results":

    st.markdown("""
    <div class="page-header">
        <h1>Inspection Results</h1>
        <p>YOLO detection output and severity assessment</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.result is None:

        if st.session_state.temp_path:
            st.info("Image loaded but not yet analysed. Go to **Road Inspection** and click **Run Detection**.")
        else:
            st.info("No analysis available. Upload an image on the **Road Inspection** page first.")

        st.stop()

    # Unpack results (unchanged)
    result               = st.session_state.result
    detections           = result["detections"]
    confirmed_detections = result["confirmed_detections"]
    possible_detections  = result["possible_detections"]
    severity             = result["severity"]
    score                = severity["score"]
    level                = severity["level"]
    reason               = severity["reason"]
    temp_path            = st.session_state.temp_path

    level_class = {
        "Severe":   "level-severe",
        "Moderate": "level-moderate",
        "Low":      "level-low",
    }.get(level, "level-none")

    # --- Summary strip ---
    st.markdown('<div class="section-heading">Summary</div>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Road Condition</div>
            <div class="stat-value {level_class}" style="font-size:22px; padding-top:4px;">{level}</div>
        </div>""", unsafe_allow_html=True)

    with sc2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Severity Score</div>
            <div class="stat-value">{score}</div>
            <div class="stat-sub">out of 10</div>
        </div>""", unsafe_allow_html=True)

    with sc3:
        total = len(detections)
        conf  = len(confirmed_detections)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Detections</div>
            <div class="stat-value">{total}</div>
            <div class="stat-sub">{conf} confirmed</div>
        </div>""", unsafe_allow_html=True)

    # --- Detection output ---
    st.markdown('<div class="section-heading">Detection Output</div>', unsafe_allow_html=True)

    img_col, det_col = st.columns([3, 2])

    with img_col:
        annotated = draw_detections(temp_path, detections)
        st.image(annotated, caption="YOLO Detection — annotated image", use_container_width=True)

        st.markdown("""
        <div style="margin-top:8px;">
            <span class="legend"><span class="legend-dot" style="background:#c0392b;"></span>Confirmed damage</span>
            <span class="legend"><span class="legend-dot" style="background:orange;"></span>Possible damage</span>
        </div>
        """, unsafe_allow_html=True)

    with det_col:

        st.markdown(f"""
        <div class="assessment-box" style="margin-bottom:14px;">
            <div class="ab-title">Assessment</div>
            <div class="ab-text">{reason}</div>
        </div>
        """, unsafe_allow_html=True)

        if confirmed_detections:

            st.markdown(
                '<div style="font-size:13px; font-weight:600; color:#1a2744; margin:14px 0 8px;">Confirmed Damage</div>',
                unsafe_allow_html=True
            )

            damage_counts = Counter(d["damage"] for d in confirmed_detections)

            for damage, count in damage_counts.items():
                confidences        = [d["confidence"] for d in confirmed_detections if d["damage"] == damage]
                highest_confidence = max(confidences)

                st.markdown(f"""
                <div class="damage-row">
                    <div>
                        <div class="dr-name">{damage}</div>
                        <div class="dr-meta">{count} instance{"s" if count > 1 else ""} detected</div>
                    </div>
                    <div class="dr-badge">{highest_confidence * 100:.0f}% conf</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="info-card" style="font-size:13px; color:#64748b; padding:12px 16px; margin-top:10px;">
                No confirmed damage detected.
            </div>""", unsafe_allow_html=True)

        if possible_detections:

            st.markdown(
                '<div style="font-size:13px; font-weight:600; color:#92400e; margin:14px 0 8px;">'
                'Possible Damage <span style="font-weight:400; color:#94a3b8;">(below confidence threshold)</span></div>',
                unsafe_allow_html=True
            )

            for det in possible_detections:
                st.markdown(f"""
                <div class="damage-row possible">
                    <div>
                        <div class="dr-name">{det["damage"]}</div>
                        <div class="dr-meta">Not included in severity score</div>
                    </div>
                    <div class="dr-badge">{det["confidence"] * 100:.0f}% conf</div>
                </div>
                """, unsafe_allow_html=True)

    # --- Detailed detections table ---
    if detections:

        st.markdown('<div class="section-heading">Detailed Detection Log</div>', unsafe_allow_html=True)

        with st.expander("View all detections with bounding-box coordinates"):

            rows = ""
            for i, d in enumerate(detections, 1):
                tag  = "Confirmed" if d["confidence"] >= 0.25 else "Possible"
                bb   = d["bounding_box"]
                rows += f"""
                <tr>
                    <td>{i}</td>
                    <td>{d["damage"]}</td>
                    <td>{d["confidence"] * 100:.1f}%</td>
                    <td>{tag}</td>
                    <td style="font-family:monospace; font-size:12px;">
                        ({bb[0]:.0f}, {bb[1]:.0f}) to ({bb[2]:.0f}, {bb[3]:.0f})
                    </td>
                </tr>"""

            st.markdown(f"""
            <table class="about-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Damage Type</th>
                        <th>Confidence</th>
                        <th>Status</th>
                        <th>Bounding Box</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run New Inspection", use_container_width=False):
        st.session_state.result    = None
        st.session_state.temp_path = None
        st.session_state.file_name = None
        st.session_state.page      = "Road Inspection"
        st.rerun()


# ============================================================
# PAGE: ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.markdown("""
    <div class="page-header">
        <h1>About Project</h1>
        <p>Final-year engineering project documentation</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <div style="font-size:14px; color:#374151; line-height:1.7;">
            <strong>RoadVision AI</strong> is a computer vision system developed as a final-year engineering project.
            It uses a fine-tuned <strong>YOLO11n</strong> object detection model to automatically identify and classify
            road surface damage from photographs, then produces a quantitative severity assessment to support road
            maintenance prioritisation decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Technical Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="about-table">
        <thead><tr><th>Component</th><th>Technology</th><th>Purpose</th></tr></thead>
        <tbody>
            <tr><td>Detection Model</td><td>YOLO11n (Ultralytics)</td><td>Real-time road damage object detection</td></tr>
            <tr><td>Deep Learning</td><td>PyTorch</td><td>Model inference backend</td></tr>
            <tr><td>Image Processing</td><td>Pillow, OpenCV</td><td>Image loading and annotation</td></tr>
            <tr><td>Web Interface</td><td>Streamlit</td><td>Interactive user interface</td></tr>
            <tr><td>Model Weights</td><td>road_damage_yolo11n_best.pt</td><td>Fine-tuned checkpoint (~5.2 MB)</td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Detection and Severity Methodology</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)

    with m1:
        st.markdown("""
        <div class="info-card">
            <div style="font-size:14px; font-weight:600; color:#1a2744; margin-bottom:8px;">Detection Pipeline</div>
            <div style="font-size:13px; color:#374151; line-height:1.7;">
                1. Image is passed to YOLO11n at a 0.10 confidence threshold to capture all candidate detections.<br>
                2. Detections above 0.25 confidence are classified as <em>Confirmed</em>.<br>
                3. Detections between 0.10 and 0.25 are flagged as <em>Possible</em> and excluded from scoring.<br>
                4. Bounding boxes are drawn on the annotated output image.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="info-card">
            <div style="font-size:14px; font-weight:600; color:#1a2744; margin-bottom:8px;">Severity Scoring</div>
            <div style="font-size:13px; color:#374151; line-height:1.7;">
                Each confirmed detection contributes a score:<br>
                <code style="background:#f1f5f9; padding:2px 6px; border-radius:3px; font-size:12px;">
                    score = weight x sqrt(confidence) x (1 + area_ratio)
                </code><br><br>
                Damage weights: Pothole = 3, Alligator Crack = 3,<br>
                Longitudinal / Transverse Crack = 2.<br>
                Area ratio accounts for the proportion of the image covered. Total score is capped at 10.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Project Structure</div>', unsafe_allow_html=True)
    st.code("""RoadVision-AI/
├── app/
│   └── app.py                        # Streamlit web interface
├── inference/
│   ├── detector.py                   # YOLO wrapper
│   ├── pipeline.py                   # Analysis pipeline
│   ├── test_image.py                 # Single-image CLI test
│   └── test_all_images.py            # Batch CLI test
├── model/
│   └── road_damage_yolo11n_best.pt   # Trained model weights
├── severity/
│   └── severity.py                   # Severity calculation logic
├── test_images/                      # Sample road images
├── notebook/
│   └── RoadVision_AI.ipynb           # Training notebook
└── requirements.txt""", language="text")

    st.markdown('<div class="section-heading">Running the Application</div>', unsafe_allow_html=True)
    st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit app
streamlit run app/app.py""", language="bash")

    st.markdown("""
    <div style="font-size:13px; color:#64748b; margin-top:8px;">
        The application runs entirely on CPU. No GPU is required.
        The trained model weights must be present at <code>model/road_damage_yolo11n_best.pt</code>.
    </div>
    """, unsafe_allow_html=True)
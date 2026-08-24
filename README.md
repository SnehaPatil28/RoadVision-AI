# RoadVision AI

### AI-Powered Road Damage Detection & Severity Assessment

RoadVision AI is a computer vision project that detects road damage from images using a YOLO-based object detection model and evaluates the overall road condition using a severity assessment system.

## Features

-  Road image upload
-  YOLO-based road damage detection
-  Bounding-box visualization
-  Detection confidence scores
-  Confirmed and possible damage detection
-  Severity score calculation
-  Road condition assessment
-  Streamlit web interface
-  Image inference and testing scripts

##  Detected Road Damage

The current model is designed to detect:

- Pothole
- Alligator Crack
- Longitudinal Crack
- Transverse Crack

##  Project Structure

```text
RoadVision-AI/
│
├── app/
│   └── app.py
│
├── inference/
│   ├── __init__.py
│   ├── detector.py
│   ├── pipeline.py
│   ├── test_image.py
│   └── test_all_images.py
│
├── model/
│   └── road_damage_yolo11n_best.pt
│
├── severity/
│   ├── __init__.py
│   └── severity.py
│
├── test_images/
│   ├── road1.jpg
│   ├── road2.jpg
│   ├── road3.jpg
│   ├── road4.jpg
│   └── road5.jpg
│
├── notebook/
│   └── RoadVision_AI.ipynb
│
├── requirements.txt
├── README.md
└── CONTRIBUTING.md

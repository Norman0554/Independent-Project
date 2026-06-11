# AI-Powered Online Exam Proctoring System

An intelligent, real-time exam monitoring system designed to ensure integrity during online tests. This application leverages computer vision, audio processing, and deep learning techniques to detect and log suspicious behaviors.

---

## Overview

This system monitors the examinee's environment, detects potential violations, provides real-time feedback, and generates comprehensive reports. 

The application runs a local computer-vision and audio capture loop, logging events to a structured file system and streaming live status updates to a web dashboard.

---

## Features

- **Face Presence Monitoring**: Detects when the face disappears from the camera frame.
- **Multi-Face Detection**: Flags an alert if more than one face is detected in the frame.
- **Eye & Gaze Tracking**: Monitors the Eye Aspect Ratio (EAR) for blink rate analysis and tracks gaze direction (Left, Right, Center) to detect if the user is looking away from the screen.
- **Mouth Movement Detection**: Analyzes lip and mouth landmarks to identify potential whispering or talking.
- **Object Detection**: Integrates a deep learning model to detect prohibited items (e.g., cell phones, textbooks, notebooks) in the camera feed.
- **Audio Monitoring**: Captures microphone input to detect voice activity. Supports speech-to-text processing for keyword flagging.
- **Dual Stream Recording**: Captures and saves video recordings of both the webcam and screen activity.
- **Real-Time Voice Alerts**: Provides audio warnings immediately upon detecting violations.
- **Web Dashboard**: Features a Flask-based administrative dashboard showing live metrics, alert logs, and a risk level indicator.
- **Automated Report Generation**: Generates detailed HTML/PDF reports at the end of the session, featuring violation timelines, heatmaps, and stats.

---

## System Architecture

```text
exam_cheating_detection/
├── config/              # Configuration files (YAML parameters)
├── models/              # Pretrained deep learning models (YOLOv8, Face Landmarker)
├── src/                 # Source code
│   ├── detection/       # Real-time detection modules (Face, Eyes, Mouth, Objects, Audio)
│   ├── reporting/       # HTML/PDF report generator
│   ├── utils/           # Helper scripts (Recorders, Alert system, Loggers)
│   ├── dashboard/       # Web-based analytics interface
│   └── main.py          # Main application loop
├── logs/                # Session log files (alerts.log)
├── recordings/          # Recorded webcam and screen videos
├── reports/             # Structured JSON data and generated HTML/PDF reports
└── tests/               # Test suites
```

---

## Technologies Used

- **Programming Language**: Python 3.8+ (Tested on Python 3.11)
- **Computer Vision**: OpenCV, MediaPipe (FaceMesh), FaceNet-PyTorch (MTCNN)
- **Deep Learning**: Ultralytics YOLOv8 (Object Detection)
- **Audio Processing**: PyAudio, OpenAI Whisper (Speech-to-Text)
- **Speech Synthesis**: gTTS (Google Text-to-Speech), Pygame Mixer
- **Data Visualization**: Matplotlib
- **Web Interface**: Flask, Jinja2
- **Document Generation**: pdfkit (wkhtmltopdf wrapper)
- **Database**: File-based storage (Structured JSON databases and plain log files)

---

## Installation & Launch

For complete installation instructions, system dependencies (such as `portaudio` and `wkhtmltopdf`), and execution commands, please refer to:
- **[README_RUN.md](file:///Users/mr.norman/Downloads/exam-cheating-detection-main/README_RUN.md)**

### Quick Start:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the proctoring client**:
   ```bash
   python src/main.py
   ```
3. **Run the dashboard** (in a separate terminal):
   ```bash
   python src/dashboard/app.py
   ```
4. **View live analytics**:
   Open `http://localhost:5000` in your web browser.

---

## Configuration

All parameters, detection sensitivity thresholds, recording FPS, and logging paths can be configured in [config/config.yaml](file:///Users/mr.norman/Downloads/exam-cheating-detection-main/config/config.yaml).

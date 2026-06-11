# Project Information (Exam Monitoring)

This document contains brief details about the programming language, technologies, database type, and overview of the project.

---

## 1. Programming Language Used
* **Python** (minimum version `3.8`, recommended version: `3.11`)

---

## 2. Frameworks and Technologies Used
* **OpenCV** — image and video processing for computer vision tasks.
* **MediaPipe** — 3D face landmarks (Face Mesh) detection for eye and mouth tracking coordinates.
* **FaceNet-PyTorch (MTCNN)** — high-accuracy face detection and recognition.
* **Ultralytics YOLOv8** — detection of prohibited objects (such as mobile phones, textbooks, notebooks).
* **Flask** — web dashboard panel showing system status in real-time.
* **PyAudio** & **OpenAI Whisper** — audio monitoring and speech-to-text conversion.
* **gTTS** (Google Text-to-Speech) & **Pygame** — real-time voice alerts and feedback.
* **pdfkit** & **Matplotlib** — statistic graphs and PDF/HTML report generation.
* **mss** — recording examinee screen activity in video format.

---

## 3. Database Type
* No dedicated SQL or NoSQL database server is used.
* **Type**: File-based storage (**File-based database**).
* All recorded violations and events are saved in JSON format inside the `reports/violations.json` file and text logs in `logs/alerts.log`.

---

## 4. Brief Project Overview
The project is an AI-powered online exam proctoring system designed to monitor candidate behavior during remote tests. The system detects the following examinee activities in real-time:
* Face presence and absence in the camera frame.
* Eye movements and gaze direction (checking if looking away from the screen).
* Mouth movement (identifying potential talking or whispering).
* Multiple face presence (flagging if more than one person is visible in frame).
* Use of prohibited items such as mobile phones and books.
* Noise levels and whispering in the environment.

The system automatically logs violations, issues verbal warnings to the candidate, records both the screen and webcam stream, and generates a visual PDF/HTML report (including a violation timeline and heatmap chart) at the end of the exam.

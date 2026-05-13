# [UNIVERSITY NAME]

## [FACULTY NAME]

## [DEPARTMENT NAME]

## Research Paper

# AI-POWERED ONLINE EXAM CHEATING DETECTION SYSTEM

Author: **MR. NORMAN**  
Scientific Adviser: **[Supervisor Name]**

**2026**

\pagebreak

# DECLARATION OF ACADEMIC INTEGRITY

Hereby I declare that this paper is my own academic work and does not contain any unacknowledged sources. The software project described in the paper was analysed, implemented, configured, and tested by the author of the paper as a student project. All external technologies, libraries, and concepts used in the project are acknowledged in the references.

Signature: ______________________

Name Surname: **Mr. Norman**

Date: **9 May 2026**

\pagebreak

# ABSTRACT

Mr. Norman "AI-Powered Online Exam Cheating Detection System". Research Paper.

Scientific Adviser: [Supervisor Name].

The paper presents the design, implementation, and analysis of an AI-powered online exam cheating detection system developed as a student software project. The system is intended to support academic integrity in remote examination environments by detecting suspicious behaviour through webcam, screen, and audio monitoring. The project combines computer vision, object detection, audio processing, real-time alerting, dashboard visualization, and automated report generation.

The aim of the paper is to examine how a multimodal monitoring system can be implemented in Python to identify common forms of suspicious activity during online examinations. The objectives include the study of online proctoring requirements, the analysis of appropriate machine learning and computer vision tools, the implementation of modular detection components, the creation of a warning and logging mechanism, and the development of a dashboard and reporting subsystem.

The methods used in the project include literature and documentation analysis, software design, modular implementation, configuration-based experimentation, manual testing, and interpretation of generated system logs and reports. The project uses OpenCV for video processing, MediaPipe for face mesh landmarks, FaceNet-PyTorch and MTCNN for face detection, YOLOv8 for prohibited object detection, PyAudio and Whisper-related tooling for audio monitoring, Flask for the dashboard, and Jinja2 with Matplotlib for report generation.

The paper is organized into three chapters. The first chapter discusses the theoretical background of online proctoring and multimodal cheating detection. The second chapter describes the architecture and implementation of the system. The third chapter presents the practical results, configuration decisions, limitations, and possible improvements. As a result of the work, the author created a working prototype that detects absent faces, multiple faces, gaze deviation, mouth movement, prohibited objects such as cell phones and books, and voice activity. The system also produces alerts, records evidence, and generates HTML reports for post-exam review.

\pagebreak

# TABLE OF CONTENTS

INTRODUCTION  
1. THEORETICAL BACKGROUND OF ONLINE EXAM CHEATING DETECTION  
1.1. Academic Integrity in Online Examination  
1.2. Computer Vision in Proctoring Systems  
1.3. Multimodal Detection of Suspicious Behaviour  
2. DESIGN AND IMPLEMENTATION OF THE SYSTEM  
2.1. General Architecture of the Project  
2.2. Face, Eye, and Mouth Detection  
2.3. Multi-Face, Object, Audio, and Screen Monitoring  
2.4. Alerts, Dashboard, and Report Generation  
3. PRACTICAL RESULTS AND SYSTEM EVALUATION  
3.1. Configuration and Runtime Environment  
3.2. Detection Results and Generated Evidence  
3.3. Limitations and Future Improvements  
THESES  
CONCLUSION  
REFERENCES  
APPENDICES

\pagebreak

# INTRODUCTION

The theme of the present paper is the development of an AI-powered online exam cheating detection system. The choice of this topic is connected with the growing importance of remote education and online assessment. When examinations are conducted outside a controlled classroom, it becomes more difficult for teachers and institutions to verify that students follow the rules of academic integrity. A student may leave the camera view, communicate with another person, use a mobile phone, read from prohibited materials, or receive help from someone nearby. These risks create a practical need for technological tools that can assist teachers in observing suspicious activity.

The project analysed in this paper is a Python-based proctoring prototype that monitors a student during an online exam. It uses webcam input, screen recording, audio monitoring, object detection, face analysis, warning messages, and report generation. The system is not intended to replace the final decision of a teacher or examiner. Instead, it is designed to collect signals that may help a human reviewer understand what happened during the exam session.

The aim of the paper is to analyse and describe the design and implementation of a multimodal AI system for detecting suspicious behaviour during online examinations.

The objectives of the paper are:

- study of the main risks that appear in online examination;
- analysis of computer vision and audio-based methods suitable for proctoring;
- implementation of separate detection modules for face presence, gaze direction, mouth movement, multiple faces, prohibited objects, screen activity, and audio activity;
- creation of a real-time alert system with a delay between repeated warnings;
- development of a dashboard that presents alerts and violations;
- generation of exam reports with violation summaries and evidence;
- evaluation of the strengths, weaknesses, and possible improvements of the prototype.

The research questions are:

- How can a Python-based system detect suspicious activity during an online examination?
- Which detection modules are most useful for identifying common cheating scenarios?
- How can warnings, logs, screenshots, recordings, and reports be organized so that the results are useful for later review?

The hypothesis of the paper is that a modular multimodal proctoring system can improve the observation of online examinations by combining webcam-based behaviour analysis, object detection, audio monitoring, and automatic reporting. The system is expected to provide more useful evidence than a single detection method because cheating behaviour can appear in different forms.

The methods of research include analysis of technical documentation and academic sources, software implementation, configuration of machine learning models, manual testing of the prototype, and interpretation of generated logs and reports. The methods of data collection include webcam frames, screen recordings, audio samples, alert logs, violation JSON files, screenshots, and generated HTML reports.

The present research is based on the study of online proctoring requirements, computer vision tools, object detection methods, and software engineering practices. The most important sources for the technical background include OpenCV documentation, MediaPipe documentation, the YOLOv8 framework, MTCNN face detection research, PyTorch documentation, and Flask documentation.

The paper contains three chapters. The first chapter explains the theoretical background of online exam cheating detection. The second chapter describes the architecture and implementation of the developed system. The third chapter presents practical results, configuration choices, limitations, and future improvements.

\pagebreak

# Chapter 1

# THEORETICAL BACKGROUND OF ONLINE EXAM CHEATING DETECTION

The present chapter focuses on the theoretical background of the project. It describes the problem of academic integrity in online examinations, explains why computer vision is useful for proctoring, and discusses the value of combining several detection methods.

## 1.1. Academic Integrity in Online Examination

Academic integrity is one of the central principles of education. In a traditional classroom examination, the examiner can observe the room, check student behaviour, and react immediately to suspicious actions. Online examinations are different because the student may be located at home or in another private environment. The examiner has limited control over the surrounding space, the devices used by the student, and the presence of other people.

The main suspicious behaviours relevant to this project are the following:

- absence of the student's face from the camera view;
- presence of more than one face in the frame;
- frequent gaze deviation from the screen;
- mouth movement that may indicate speaking;
- appearance of prohibited objects such as a mobile phone or book;
- voice activity in the examination environment;
- screen activity that should be recorded for later review.

These behaviours do not always prove cheating. For example, a student may look away because of distraction, or a phone may appear in the background without being used. Therefore, the system should not be treated as a final judge. It should be understood as an evidence collection and warning tool. The final interpretation should remain the responsibility of a human reviewer.

From a software engineering point of view, online proctoring requires real-time processing, low delay, reliable logging, and understandable reporting. A warning that is generated too often may become annoying and reduce the usability of the system. For that reason, the project includes a warning delay mechanism that reduces repeated alerts for the same violation type.

## 1.2. Computer Vision in Proctoring Systems

Computer vision is the main technical basis of the project. It allows the system to process webcam frames and extract information about the student's face, eyes, mouth, and surrounding objects. The project uses OpenCV as the main tool for video capture, frame manipulation, drawing text on frames, and recording video output.

Face detection is necessary because the system must know whether the student is visible. If the face disappears from the frame, the system marks it as a suspicious event. The project uses FaceNet-PyTorch and MTCNN to detect faces. MTCNN is suitable for this task because it can detect faces and return confidence values. These confidence values are important because low-confidence detections may produce false alerts.

Eye and gaze analysis are implemented with MediaPipe Face Mesh. MediaPipe provides facial landmarks that describe the position of different parts of the face. The system uses eye landmarks to calculate an approximate eye aspect ratio and to determine gaze direction. The gaze direction is interpreted as center, left, or right. If the gaze is not centered, the system can warn the student to keep looking at the screen.

Mouth monitoring is also based on facial landmarks. The system analyses the distance between upper and lower lip points and the width of the mouth. If the mouth is open or moving for several frames, the system treats it as possible speaking. This detection is not perfect because normal facial expressions may also move the mouth. However, when combined with audio detection, it becomes a useful additional signal.

Object detection is performed with YOLOv8. In the project, the most important prohibited objects are cell phones and books. YOLOv8 can identify object classes in a frame and return a confidence score. The project was adjusted to detect phones by class name, such as "cell phone", instead of relying only on a class ID. This makes the detection logic clearer and more robust.

## 1.3. Multimodal Detection of Suspicious Behaviour

A single detection method is not enough for reliable proctoring. Face detection can identify whether the student is visible, but it cannot know whether the student is speaking. Gaze detection can show that the student looks away, but it cannot prove that the student is reading notes. Object detection can identify a phone, but it may not know whether the phone is being used. For this reason, the project uses a multimodal approach.

The system combines the following modalities:

- visual face monitoring;
- gaze monitoring;
- mouth movement monitoring;
- multiple face detection;
- object detection;
- screen recording;
- audio monitoring;
- real-time warnings;
- dashboard statistics;
- post-session reports.

The advantage of this approach is that each module provides a different kind of evidence. If several modules detect suspicious behaviour at the same time, the event becomes more significant. For example, if the student looks away, mouth movement is detected, and voice activity is recorded, the case may require closer human review.

Thus, the theoretical background shows that online proctoring should be understood as a combination of observation, evidence collection, and human interpretation. The following chapter describes how this idea was implemented in the project.

\pagebreak

# Chapter 2

# DESIGN AND IMPLEMENTATION OF THE SYSTEM

The present chapter describes the architecture and implementation of the developed system. It explains how the project is organized, which modules are responsible for detection, and how alerts, dashboard data, and reports are generated.

## 2.1. General Architecture of the Project

The project is organized as a modular Python application. The main entry point is `src/main.py`. It loads configuration from `config/config.yaml`, initializes detection modules, starts recording systems, processes webcam frames, displays the results in a preview window, logs violations, and generates a report at the end of the session.

The project structure is divided into several directories:

- `config` contains the YAML configuration file;
- `src/detection` contains detection modules;
- `src/utils` contains logging, recording, alerting, and capture utilities;
- `src/dashboard` contains the Flask dashboard;
- `src/reporting` contains report generation logic and report templates;
- `logs`, `recordings`, and `reports` store runtime output.

Table 1 presents the main system modules and their responsibilities.

Table 1. Main modules of the project

| Module | Responsibility |
| --- | --- |
| `main.py` | Coordinates configuration, detection, alerting, recording, and reporting |
| `face_detection.py` | Detects whether the student's face is present |
| `eye_tracking.py` | Tracks eye landmarks and estimates gaze direction |
| `mouth_detection.py` | Detects mouth movement that may indicate speaking |
| `multi_face.py` | Detects if more than one face appears in the frame |
| `object_detection.py` | Detects prohibited objects such as phones and books |
| `audio_detection.py` | Detects voice activity and optionally processes speech |
| `screen_capture.py` | Records screen activity |
| `alert_system.py` | Provides voice warnings when audio playback is available |
| `violation_logger.py` | Saves structured violation information |
| `dashboard/app.py` | Displays live dashboard data through Flask |
| `report_generator.py` | Produces HTML or PDF-style reports after the session |

The configuration file is important because it allows the author of the paper to adjust the behaviour of the system without changing the source code. For example, the configuration includes video resolution, frame rate, detection thresholds, object detection confidence, audio monitoring settings, log paths, and warning delay values.

## 2.2. Face, Eye, and Mouth Detection

The face detection module uses MTCNN through FaceNet-PyTorch. The module reads a frame, converts it from BGR to RGB, and applies the detector. If at least one face is detected with sufficient confidence, the system considers the student visible. If no face is detected, the system produces the `FACE_DISAPPEARED` violation after the configured delay.

Eye tracking is implemented with MediaPipe Face Mesh. The module initializes the Face Mesh model and uses a set of landmark indices for both eyes. It calculates the eye aspect ratio and estimates gaze direction by comparing the horizontal position of the eye centers with the nose tip. If the gaze direction is not center, the main loop sets a warning flag and may log `GAZE_AWAY`.

Mouth detection also uses MediaPipe landmarks. The module compares the vertical distance between upper and lower lip landmarks and the mouth width. If the mouth is open or moving for several consecutive frames, the system marks the mouth state as moving. This produces a `MOUTH_MOVING` violation when the warning delay has passed.

The face, eye, and mouth modules are closely connected because they all depend on visible facial data. Their output is displayed on the webcam preview. Normal values are drawn in green, while warning states are drawn in red. This visual feedback is important because the student can immediately understand which behaviour is considered suspicious.

## 2.3. Multi-Face, Object, Audio, and Screen Monitoring

The multi-face detector uses MTCNN with `keep_all=True`. It checks how many high-confidence faces appear in the frame. If two or more faces are detected for a configured number of consecutive frames, the system logs `MULTIPLE_FACES`. This is one of the most serious warning types because another person near the student may indicate outside help.

Object detection uses YOLOv8. The system loads the `yolov8n.pt` model and searches for prohibited objects. The most important classes are `cell phone` and `book`. During development, phone detection was improved by checking the class name returned by YOLO rather than depending only on a class number. The configuration also includes a lower confidence threshold for phones because mobile phones may be small in the frame and harder to detect.

Audio monitoring is implemented with PyAudio and NumPy. The system reads audio chunks, calculates energy and zero-crossing rate, and marks voice activity when the signal passes the configured thresholds. Whisper can be enabled for additional speech processing, although it is disabled by default to reduce computational load. The audio system can produce voice-related alerts, but it is also designed to continue running even when audio output is not available.

Screen monitoring is implemented with MSS and OpenCV video recording. It captures the selected monitor and writes frames to an MP4 file. During development, the screen recorder was made more robust by checking monitor availability and preventing crashes when the monitor index or size is invalid.

## 2.4. Alerts, Dashboard, and Report Generation

The alert subsystem has two parts: logging and optional voice output. The logger writes alert messages to `logs/alerts.log`. The voice alert system uses gTTS and Pygame when audio playback is available. If Pygame mixer cannot be initialized, the system disables voice alerts and continues working. This design is important because audio problems should not stop the whole proctoring session.

The main loop uses a warning delay of approximately five seconds. This delay prevents the system from creating too many repeated warnings for the same violation type. The warning delay is especially important for mouth movement and gaze deviation because these events can happen frequently during normal behaviour.

The dashboard is implemented with Flask. It provides a web interface at `http://localhost:5000`. The dashboard reads alert logs and violation files, displays recent events, shows total alerts and total violations, and estimates a simple cheating probability based on the number of violations. The dashboard is not a statistical proof of cheating. It is a monitoring interface for quickly reviewing system activity.

The report generator uses Jinja2 templates and Matplotlib. At the end of a session, it prepares a report with student information, violation statistics, a timeline, and a heatmap when data are available. The system can generate an HTML report when PDF generation through `wkhtmltopdf` is not available. This fallback improves reliability on machines where PDF tools are not installed.

Thus, the implementation follows a modular structure. Each detection module solves one specific problem, while the main program combines these results into warnings, logs, recordings, dashboard data, and reports.

\pagebreak

# Chapter 3

# PRACTICAL RESULTS AND SYSTEM EVALUATION

The present chapter describes the practical results of the project. It explains how the system was configured, what evidence it generates, and which limitations were observed during development and testing.

## 3.1. Configuration and Runtime Environment

The project was configured to run in a Python 3.11 virtual environment. Python 3.14 caused compatibility problems with MediaPipe and Pygame, therefore the environment was rebuilt with Python 3.11. The dependency list was updated so that MediaPipe uses a compatible version. The system also requires the YOLOv8 nano model, which is stored in the `models` directory.

The most important configuration values are presented in Table 2.

Table 2. Main configuration values

| Configuration area | Value used in the project | Purpose |
| --- | --- | --- |
| Video source | `0` | Default webcam |
| Video resolution | `1280 x 720` | Webcam frame size |
| Video FPS | `30` | Webcam recording rate |
| Screen FPS | `15` | Screen recording rate |
| Face confidence | `0.8` | Minimum face confidence |
| Object confidence | `0.45` | General object threshold |
| Phone confidence | `0.35` | Lower threshold for phone detection |
| Warning delay | `5 seconds` | Prevents repeated warning spam |
| Audio monitoring | Enabled | Voice activity detection |
| Dashboard port | `5000` | Web monitoring interface |

The system can be started with the command:

`python src/main.py`

The dashboard can be started in a separate terminal with:

`python src/dashboard/app.py`

The author of the paper observed that runtime stability depends on the operating system, camera permissions, microphone permissions, and screen recording permissions. On macOS, SDL and audio warnings may appear because of interactions between OpenCV and Pygame. These warnings are not always fatal, but they should be documented because they affect the user experience.

## 3.2. Detection Results and Generated Evidence

The prototype produces several kinds of output. First, it displays a live webcam preview with status text. Normal states are green. Warning states are red. This gives immediate feedback to the student. For example, if the student's gaze is not centered, the preview displays a red warning asking the student to keep the gaze centered.

Second, the system writes alert logs. These logs are simple text entries that include time, alert type, and message. They are useful for the dashboard because the dashboard can read the log file and display recent alerts.

Third, the system writes structured violation data into JSON format. Structured data are important because they can be processed later by the dashboard or report generator. Each violation contains the type, timestamp, and metadata.

Fourth, the system captures evidence images. When a violation is detected, the current webcam frame can be saved as a screenshot. This is useful because a human reviewer can later see the visual context of the alert.

Fifth, the system records webcam and screen sessions. These recordings provide additional evidence for later review. They also allow the examiner to compare warning logs with actual video behaviour.

Sixth, the system generates reports. The generated report contains the student's information, a list of violations, statistics, and visual summaries such as a timeline or heatmap when data are available. The report is saved in the `reports/generated` directory.

The most successful parts of the prototype are face detection, multiple face detection, dashboard data reading, violation logging, and report generation. Phone detection required additional configuration because phones are often small objects in the frame. The confidence threshold for phone detection was lowered and the inference width was increased to improve detection.

## 3.3. Limitations and Future Improvements

The system is a working prototype, but it has several limitations. The first limitation is that some detections may produce false positives. For example, a student may move the mouth without speaking, or look away for a harmless reason. The system should therefore be used as an assistant, not as an automatic punishment tool.

The second limitation is hardware dependence. Webcam quality, lighting, microphone quality, and CPU performance strongly influence detection accuracy. Low light may reduce face and landmark detection quality. Slow hardware may reduce the frame rate and delay alerts.

The third limitation is object detection accuracy. YOLOv8 can detect phones and books, but detection depends on object size, angle, occlusion, and lighting. A phone held partly outside the frame may not be detected. Further improvement may require a custom dataset trained specifically on exam scenarios.

The fourth limitation is the simple nature of the cheating probability score in the dashboard. At present, it is based on the number of violations. A more advanced system should consider violation type, duration, frequency, and combinations of signals.

The fifth limitation is privacy. A system that records webcam, screen, and audio data must be used carefully. Students should be informed about what data are collected, where they are stored, who can access them, and how long they are kept.

Future improvements may include:

- training a custom object detection model for exam environments;
- adding real-time video stream to the dashboard;
- improving gaze estimation with iris landmarks;
- distinguishing normal mouth movement from speech more accurately;
- adding role-based access to the dashboard;
- generating PDF reports without external system dependencies;
- adding unit tests and integration tests;
- improving privacy controls and data retention settings.

Thus, the practical evaluation shows that the project is technically functional and useful as a prototype. At the same time, the results should be interpreted carefully because proctoring decisions have academic and ethical consequences.

\pagebreak

# THESES

1. Online examinations create academic integrity risks because the examiner cannot fully control the student's physical environment.

2. A useful proctoring system should collect evidence and support human review rather than automatically declare a student guilty of cheating.

3. Face detection is necessary for verifying that the student remains visible during the examination.

4. Gaze detection helps identify moments when the student looks away from the screen, although this signal should be interpreted cautiously.

5. Mouth movement detection can indicate possible speaking, especially when combined with audio monitoring.

6. Multi-face detection is one of the most important warning types because another person in the frame may suggest external assistance.

7. Object detection with YOLOv8 can identify prohibited objects such as cell phones and books, but phone detection requires careful confidence thresholds because phones may appear small in the frame.

8. Warning delays are necessary because repeated alerts can distract the student and reduce the usability of the system.

9. Dashboard visualization improves the practical value of the system by presenting alerts, violations, and statistics in one place.

10. Automated report generation is important because it converts raw logs and screenshots into a structured form suitable for later review.

11. The prototype demonstrates that a modular Python system can combine computer vision, audio monitoring, logging, dashboard visualization, and reporting for online exam proctoring.

12. Further development should focus on better accuracy, privacy, usability, and custom training data for exam-specific object detection.

\pagebreak

# CONCLUSION

Having done the research into the development of an AI-powered online exam cheating detection system, the author of the paper has come to the following conclusions.

Online exam proctoring is a relevant and practical topic because remote education requires methods for supporting academic integrity. The problem is not limited to detecting one specific behaviour. A student may leave the camera view, look away from the screen, speak to another person, use a phone, read from a book, or receive help from someone nearby. Therefore, a proctoring system should combine several detection methods.

In this project the author implemented a modular Python prototype that combines face detection, gaze tracking, mouth movement detection, multiple face detection, object detection, audio monitoring, screen recording, real-time alerts, dashboard visualization, and report generation. The system uses established libraries such as OpenCV, MediaPipe, FaceNet-PyTorch, YOLOv8, PyAudio, Flask, Jinja2, and Matplotlib.

The practical work showed that modular architecture is suitable for this type of system. Each module can be developed, configured, and improved separately. The main program is responsible for coordinating the modules and collecting their results into a single workflow. This architecture also makes debugging easier because errors in one subsystem can be isolated without rewriting the whole project.

The project also showed the importance of configuration. Detection thresholds, warning delays, audio settings, and object detection parameters must be adjustable. For example, phone detection required a lower confidence threshold and a larger inference size because a phone is usually a small object in the camera frame.

During development several technical difficulties were encountered. The Python version had to be changed to Python 3.11 because of compatibility issues with MediaPipe. Voice alerts had to be made optional because audio playback may fail on some systems. Screen recording had to be protected from invalid monitor configurations. These difficulties were useful because they showed that a practical system must handle real operating system differences, not only theoretical algorithms.

The generated dashboard and reports make the system more useful for exam review. Alerts alone are not enough; they must be organized and presented clearly. The dashboard gives a quick overview during or after a session, while the report provides structured evidence for later analysis.

The author realizes that the prototype should not be used as an automatic decision-making system. It may produce false positives and false negatives. Human review remains necessary. The system should be treated as a support tool that collects evidence and indicates suspicious moments.

For further research, the author recommends creating an exam-specific dataset, improving gaze estimation, training a custom object detector, adding stronger privacy controls, and evaluating the system with more users and more realistic exam scenarios. The present project may be useful for students and teachers interested in educational technology, computer vision, and the practical implementation of academic integrity tools.

\pagebreak

# REFERENCES

Bradski, G. (2000) The OpenCV Library. Dr. Dobb's Journal of Software Tools.

Flask Documentation (2026) Flask Web Development Framework. Available at: https://flask.palletsprojects.com/ (Accessed: 9 May 2026).

Jocher, G., Chaurasia, A. and Qiu, J. (2023) Ultralytics YOLO. Available at: https://github.com/ultralytics/ultralytics (Accessed: 9 May 2026).

Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C., Yong, M., Lee, J., Chang, W., Hua, W., Georg, M. and Grundmann, M. (2019) MediaPipe: A Framework for Building Perception Pipelines. Available at: https://arxiv.org/abs/1906.08172 (Accessed: 9 May 2026).

OpenCV Documentation (2026) Open Source Computer Vision Library. Available at: https://docs.opencv.org/ (Accessed: 9 May 2026).

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L. et al. (2019) PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems 32.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C. and Sutskever, I. (2022) Robust Speech Recognition via Large-Scale Weak Supervision. Available at: https://arxiv.org/abs/2212.04356 (Accessed: 9 May 2026).

Sardiko, L. (2004) Guidelines on Writing a Term Paper, a Bachelor Paper, a Master Paper. Daugavpils: Daugavpils University.

Zhang, K., Zhang, Z., Li, Z. and Qiao, Y. (2016) Joint Face Detection and Alignment Using Multi-task Cascaded Convolutional Networks. IEEE Signal Processing Letters. Vol. 23/10.

\pagebreak

# APPENDICES

## Appendix 1. System Run Commands

Main detection system:

`source venv/bin/activate`

`python src/main.py`

Dashboard:

`source venv/bin/activate`

`python src/dashboard/app.py`

Dashboard URL:

`http://localhost:5000`

## Appendix 2. Main Project Outputs

The project produces the following outputs:

- webcam recordings in `recordings`;
- screen recordings in `recordings`;
- alert log in `logs/alerts.log`;
- violation data in `reports/violations.json`;
- violation screenshots in `reports/violation_captures`;
- generated reports in `reports/generated`;
- dashboard data through Flask API routes.

## Appendix 3. Main Violation Types

| Violation type | Meaning |
| --- | --- |
| `FACE_DISAPPEARED` | The student's face is not visible |
| `GAZE_AWAY` | The student's gaze is not centered |
| `MOUTH_MOVING` | The student's mouth movement may indicate speaking |
| `MULTIPLE_FACES` | More than one person is visible |
| `OBJECT_DETECTED` | A prohibited object such as a phone or book is visible |
| `VOICE_DETECTED` | Voice activity is detected in the environment |

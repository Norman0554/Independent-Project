import cv2
import yaml
from datetime import datetime
import time
from detection.face_detection import FaceDetector
from detection.eye_tracking import EyeTracker
from detection.mouth_detection import MouthMonitor
from detection.object_detection import ObjectDetector
from detection.multi_face import MultiFaceDetector
from detection.audio_detection import AudioMonitor
from utils.video_utils import VideoRecorder
from utils.screen_capture import ScreenRecorder
from utils.logging import AlertLogger
from utils.alert_system import AlertSystem
from utils.violation_logger import ViolationLogger
from utils.screenshot_utils import ViolationCapturer
from reporting.report_generator import ReportGenerator


def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)

def display_detection_results(frame, results):
    y_offset = 30
    line_height = 30

    warning_flags = results.get('warning_flags', {})

    status_items = [
        (f"Face: {'Present' if results['face_present'] else 'Absent'}", warning_flags.get('face', False)),
        (f"Gaze: {results['gaze_direction']}", warning_flags.get('gaze', False)),
        (f"Eyes: {'Open' if results['eye_ratio'] > 0.25 else 'Closed'}", False),
        (f"Mouth: {'Moving' if results['mouth_moving'] else 'Still'}", warning_flags.get('mouth', False))
    ]

    alert_items = []
    if warning_flags.get('face', False):
        alert_items.append("WARNING: Face Not Visible")
    if warning_flags.get('gaze', False):
        alert_items.append("WARNING: Keep Gaze Center")
    if warning_flags.get('mouth', False):
        alert_items.append("WARNING: Mouth Movement Detected")
    if warning_flags.get('multiple_faces', False):
        alert_items.append("WARNING: Multiple Faces Detected")
    if warning_flags.get('objects', False):
        alert_items.append("WARNING: Suspicious Object Detected")

    for item, is_warning in status_items:
        color = (0, 0, 255) if is_warning else (0, 255, 0)
        cv2.putText(frame, item, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += line_height

    for item in alert_items:
        cv2.putText(frame, item, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += line_height

    cv2.putText(frame, results['timestamp'], 
               (frame.shape[1] - 250, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def main():
    config = load_config()
    alert_logger = AlertLogger(config)
    alert_system = AlertSystem(config)
    violation_capturer = ViolationCapturer(config)
    violation_logger = ViolationLogger(config)
    report_generator = ReportGenerator(config)

    student_info = {
        'id': 'STUDENT_001',
        'name': 'John Doe',
        'exam': 'Final Examination',
        'course': 'Computer Science 101'
    }

    
    # Initialize recorders
    video_recorder = VideoRecorder(config)
    screen_recorder = ScreenRecorder(config)
    
    # Initialize audio monitor
    audio_monitor = AudioMonitor(config)
    audio_monitor.alert_system = alert_system
    audio_monitor.alert_logger = alert_logger

    audio_started = False
    screen_started = False
    video_started = False
    cap = None
    warning_delay_seconds = config.get('logging', {}).get('warning_delay_seconds', 5)
    last_violation_time = {}

    audio_cfg = config.get('detection', {}).get('audio_monitoring', {})
    if audio_cfg.get('enabled', False):
        try:
            audio_monitor.start()
            audio_started = True
        except Exception as e:
            print(f"Audio monitoring disabled: {e}")

    try:
        if config['screen']['recording']:
            try:
                screen_recorder.start_recording()
                screen_started = True
            except Exception as e:
                print(f"Screen recording disabled: {e}")
        # Initialize detectors
        detectors = [
            FaceDetector(config),
            EyeTracker(config),
            MouthMonitor(config),
            MultiFaceDetector(config),
            ObjectDetector(config),
        ]
        
        for detector in detectors:
            if hasattr(detector, 'set_alert_logger'):
                detector.set_alert_logger(alert_logger)

        cap = cv2.VideoCapture(config['video']['source'])
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {config['video']['source']}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['video']['resolution'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['video']['resolution'][1])

        # Start webcam recording after confirming that the camera is available.
        try:
            video_recorder.start_recording()
            video_started = True
        except Exception as e:
            print(f"Webcam recording disabled: {e}")

        def trigger_violation(violation_type, frame_data, metadata):
            now = time.time()
            last = last_violation_time.get(violation_type, 0)
            if (now - last) < warning_delay_seconds:
                return False

            last_violation_time[violation_type] = now
            alert_system.speak_alert(violation_type)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            violation_capturer.capture_violation(frame_data, violation_type, timestamp)
            violation_logger.log_violation(violation_type, timestamp, metadata)
            return True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = {
                'face_present': False,
                'gaze_direction': 'Center',
                'eye_ratio': 0.3,
                'mouth_moving': False,
                'multiple_faces': False,
                'objects_detected': False,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Perform detections
            results['face_present'] = detectors[0].detect_face(frame)
            results['gaze_direction'], results['eye_ratio'] = detectors[1].track_eyes(frame)
            results['mouth_moving'] = detectors[2].monitor_mouth(frame)
            results['multiple_faces'] = detectors[3].detect_multiple_faces(frame)
            results['objects_detected'] = detectors[4].detect_objects(frame)
            gaze_direction_normalized = str(results['gaze_direction']).strip().lower()
            gaze_is_center = gaze_direction_normalized == "center"

            results['warning_flags'] = {
                'face': not results['face_present'],
                'gaze': not gaze_is_center,
                'mouth': results['mouth_moving'],
                'multiple_faces': results['multiple_faces'],
                'objects': results['objects_detected'],
            }

            if results['warning_flags']['face']:
                trigger_violation(
                    "FACE_DISAPPEARED",
                    frame,
                    {'duration': f'{warning_delay_seconds}+ seconds', 'frame': results}
                )
            elif results['warning_flags']['multiple_faces']:
                trigger_violation(
                    "MULTIPLE_FACES",
                    frame,
                    {'duration': f'{warning_delay_seconds}+ seconds', 'frame': results}
                )
            elif results['objects_detected']:
                trigger_violation(
                    "OBJECT_DETECTED",
                    frame,
                    {'duration': f'{warning_delay_seconds}+ seconds', 'frame': results}
                )
            elif results['warning_flags']['gaze']:
                trigger_violation(
                    "GAZE_AWAY",
                    frame,
                    {'duration': f'{warning_delay_seconds}+ seconds', 'frame': results}
                )
            elif results['warning_flags']['mouth']:
                trigger_violation(
                    "MOUTH_MOVING",
                    frame,
                    {'duration': f'{warning_delay_seconds}+ seconds', 'frame': results}
                )

            # Display and record
            display_detection_results(frame, results)
            video_recorder.record_frame(frame)
            
            # Show preview
            cv2.imshow('Exam Proctoring', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        if audio_started:
            audio_monitor.stop()

        violations = violation_logger.get_violations()
        report_path = report_generator.generate_report(student_info, violations)
        print(f"Report generated: {report_path}")
        if screen_started:
            try:
                screen_data = screen_recorder.stop_recording()
                if screen_data:
                    print(f"Screen recording saved: {screen_data['filename']}")
            except Exception as e:
                print(f"Screen recording failed: {e}")

        if video_started:
            try:
                video_data = video_recorder.stop_recording()
                if video_data:
                    print(f"Webcam recording saved: {video_data['filename']}")
            except Exception as e:
                print(f"Webcam recording failed: {e}")
        
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

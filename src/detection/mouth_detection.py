from detection.face_landmarks import FaceLandmarkDetector, get_landmark

class MouthMonitor:
    def __init__(self, config):
        self.face_landmarks = FaceLandmarkDetector(max_num_faces=1)
            
        self.mouth_threshold = config['detection']['mouth']['movement_threshold']
        self.mouth_movement_count = 0
        self.last_mouth_time = None
        self.alert_logger = None  # Will be set externally
        
    def set_alert_logger(self, alert_logger):
        self.alert_logger = alert_logger
        
    def monitor_mouth(self, frame):
        faces = self.face_landmarks.process(frame)
        
        if not faces:
            return False
            
        face_landmarks = faces[0]
        
        # Get mouth landmarks (using more points for better accuracy)
        mouth_points = [
            13,  # Upper inner lip
            14,  # Lower inner lip
            78,  # Right corner
            306,  # Left corner
            312,  # Upper outer lip
            317,  # Lower outer lip
        ]
        
        # Calculate mouth openness
        upper_lip = get_landmark(face_landmarks, 13).y
        lower_lip = get_landmark(face_landmarks, 14).y
        mouth_open = lower_lip - upper_lip
        
        # Calculate mouth width
        right_corner = get_landmark(face_landmarks, 78).x
        left_corner = get_landmark(face_landmarks, 306).x
        mouth_width = abs(right_corner - left_corner)
        
        if mouth_open > 0.03 or mouth_width > 0.2:  # Thresholds for mouth movement
            self.mouth_movement_count += 1
            
            if self.mouth_movement_count > self.mouth_threshold and self.alert_logger:
                self.alert_logger.log_alert(
                    "MOUTH_MOVEMENT", 
                    "Excessive mouth movement detected (possible talking)"
                )
                self.mouth_movement_count = 0
            return True
        else:
            self.mouth_movement_count = max(0, self.mouth_movement_count - 1)
            return False

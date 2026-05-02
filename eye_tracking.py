import cv2
import dlib
import numpy as np
import os
from collections import deque

class EyeTracker:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        if not os.path.exists(predictor_path):
            raise FileNotFoundError("Please ensure shape_predictor_68_face_landmarks.dat is in the folder.")
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Thresholds (Adjustable in config.json later)
        self.left_thr = 0.65
        self.right_thr = 0.35
        self._ratio_buf = deque(maxlen=5)
        self.calibrating = False # Always ready

    def _get_eye_ratio(self, eye_points, gray):
        # Create a mask for the eye shape
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [eye_points], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Crop the eye area
        min_x = np.min(eye_points[:, 0])
        max_x = np.max(eye_points[:, 0])
        min_y = np.min(eye_points[:, 1])
        max_y = np.max(eye_points[:, 1])
        
        eye_crop = gray[min_y:max_y, min_x:max_x]
        _, thresh = cv2.threshold(eye_crop, 70, 255, cv2.THRESH_BINARY_INV)
        
        h, w = thresh.shape
        left_side = thresh[0:h, 0:int(w/2)]
        right_side = thresh[0:h, int(w/2):w]
        
        left_white = cv2.countNonZero(left_side)
        right_white = cv2.countNonZero(right_side)
        
        if left_white == 0: return 0.5
        if right_white == 0: return 0.5
        
        return left_white / (left_white + right_white)

    def get_gaze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return "NO_EYE", False, frame

        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Left eye landmarks (36 to 41)
        eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        
        # Draw eye outline
        cv2.polylines(frame, [eye_pts], True, (0, 255, 0), 1)
        
        ratio = self._get_eye_ratio(eye_pts, gray)
        self._ratio_buf.append(ratio)
        smooth_ratio = sum(self._ratio_buf) / len(self._ratio_buf)
        
        if smooth_ratio > self.left_thr:
            gaze_dir = "LEFT"
        elif smooth_ratio < self.right_thr:
            gaze_dir = "RIGHT"
        else:
            gaze_dir = "FORWARD"
            
        cv2.putText(frame, f"Gaze: {gaze_dir} ({smooth_ratio:.2f})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return gaze_dir, True, frame

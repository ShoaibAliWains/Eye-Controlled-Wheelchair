import cv2
import numpy as np
from collections import deque

class EyeTracker:
    def __init__(self):
        # Thresholds for LEFT and RIGHT based on screen width percentage
        self.left_thr = 0.40
        self.right_thr = 0.60
        self._ratio_buf = deque(maxlen=5)
        print("[EyeTracker] Direct Pupil Blob Tracking Initialized (Video Style) 🚀")

    def get_gaze(self, frame):
        h, w = frame.shape[:2]
        display = frame.copy()
        
        # 1. Convert to Grayscale and Blur to remove noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)

        # 2. Find the darkest 3% of pixels (This will be the pupil)
        dark_level = max(10, int(np.percentile(blur, 3)))
        _, thresh = cv2.threshold(blur, dark_level, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up eyelashes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # 3. Find Contours (Blobs)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        best_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter tiny noise and massive shadows
            if 100 < area < (w * h * 0.5):
                if area > best_area:
                    best_area = area
                    best_cnt = cnt

        # Convert threshold image to BGR so we can overlay it on HUD
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        if best_cnt is None:
            cv2.putText(display, "PUPIL LOST", (w//2 - 70, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return "NO_EYE", thresh_color, display

        # 4. Calculate Center of the Pupil
        M = cv2.moments(best_cnt)
        if M["m00"] == 0:
            return "NO_EYE", thresh_color, display
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 5. Determine Gaze Direction
        ratio = cx / w
        self._ratio_buf.append(ratio)
        smooth_ratio = sum(self._ratio_buf) / len(self._ratio_buf)

        if smooth_ratio < self.left_thr:
            gaze_dir = "LEFT"
            color = (0, 165, 255)
        elif smooth_ratio > self.right_thr:
            gaze_dir = "RIGHT"
            color = (0, 165, 255)
        else:
            gaze_dir = "FORWARD"
            color = (0, 255, 0)

        # 6. Draw HUD (Crosshair and Boundaries)
        cv2.drawContours(display, [best_cnt], -1, color, 2)
        cv2.line(display, (cx - 15, cy), (cx + 15, cy), color, 2)
        cv2.line(display, (cx, cy - 15), (cx, cy + 15), color, 2)
        
        lx = int(w * self.left_thr)
        rx = int(w * self.right_thr)
        cv2.line(display, (lx, 0), (lx, h), (255, 0, 0), 2)
        cv2.line(display, (rx, 0), (rx, h), (255, 0, 0), 2)
        
        cv2.putText(display, f"Ratio: {smooth_ratio:.2f}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return gaze_dir, thresh_color, display

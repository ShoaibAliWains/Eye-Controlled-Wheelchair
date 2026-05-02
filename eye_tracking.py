import cv2
import numpy as np
from collections import deque

class EyeTracker:
    def __init__(self):
        # HARDCODED THRESHOLDS - NO CALIBRATION NEEDED
        self.left_thr = 0.45
        self.right_thr = 0.55
        
        # Generous Pupil Area Fractions
        self._min_frac = 0.01 
        self._max_frac = 0.80
        self._dark_pct = 35
        
        # Original Haar Cascade (Checks for Eye shape, ignores hair)
        self._cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        
        self._ratio_buf = deque(maxlen=8)
        self._last_eye_box = None
        self.calibrating = False
        
        print("[EyeTracker] Original Eye-Focus AI Restored + Calibration Bypassed 🚀")

    def get_gaze(self, frame):
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # 1. FIND THE EYE (maxSize increased so it works close up, ignores hair)
        eyes = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(25, 15), maxSize=(600, 400))
        
        eye_box = None
        if len(eyes) > 0:
            # Get the best valid eye box
            valid = [b for b in eyes if b[2] / max(b[3], 1) > 1.1]
            if valid:
                eye_box = tuple(sorted(valid, key=lambda b: b[2] * b[3], reverse=True)[0])

        if eye_box is None:
            eye_box = self._last_eye_box

        if eye_box is None:
            h, w = display.shape[:2]
            cv2.putText(display, "NO EYE DETECTED", (w // 2 - 130, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2)
            return "NO_EYE", False, display

        self._last_eye_box = eye_box
        ex, ey, ew, eh = eye_box

        # Draw Cyan Box around the valid EYE
        cv2.rectangle(display, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 2)

        # 2. FIND PUPIL INSIDE THE EYE BOX ONLY (Hair outside is ignored)
        roi = gray[ey:ey + eh, ex:ex + ew]
        
        dark_level = int(np.percentile(roi, self._dark_pct))
        _, thresh = cv2.threshold(roi, dark_level, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_cnt = None
        best_score = -1
        box_area = ew * eh
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (box_area * self._min_frac < area < box_area * self._max_frac):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = 4 * np.pi * area / (perim ** 2)
            if circ < 0.2: 
                continue
            
            score = area * circ
            if score > best_score:
                best_score = score
                best_cnt = cnt

        if best_cnt is None:
            return "FORWARD", True, display

        # 3. CALCULATE GAZE DIRECTION
        M = cv2.moments(best_cnt)
        if M["m00"] == 0:
            return "FORWARD", True, display
            
        px = int(M["m10"] / M["m00"])
        py = int(M["m01"] / M["m00"])
        
        ratio = px / max(ew, 1)
        self._ratio_buf.append(ratio)
        smooth = sum(self._ratio_buf) / len(self._ratio_buf)

        if smooth < self.left_thr:
            gaze_dir = "LEFT"
        elif smooth > self.right_thr:
            gaze_dir = "RIGHT"
        else:
            gaze_dir = "FORWARD"

        # 4. DRAW HUD
        abs_cx, abs_cy = ex + px, ey + py
        shifted = best_cnt + np.array([[[ex, ey]]])
        cv2.drawContours(display, [shifted], -1, (0, 255, 80), 2)
        cv2.circle(display, (abs_cx, abs_cy), 4, (0, 255, 80), -1)

        lx = int(ex + ew * self.left_thr)
        rx = int(ex + ew * self.right_thr)
        cv2.line(display, (lx, ey), (lx, ey + eh), (255, 150, 0), 1)
        cv2.line(display, (rx, ey), (rx, ey + eh), (255, 150, 0), 1)
        
        # Add label above eye
        cmd_col = (0, 255, 80) if gaze_dir == "FORWARD" else (0, 165, 255)
        cv2.putText(display, gaze_dir, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cmd_col, 2)

        return gaze_dir, True, display

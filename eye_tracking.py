import cv2
import numpy as np

class EyeTracker:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        
        # Temporal smoothing for jittery tracking
        self.smooth_x = None
        self.smooth_y = None
        self.alpha = 0.4 # Smoothing factor (0.0 to 1.0)

    def get_pupil_position(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5) 

        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

        if len(eyes) > 0:
            eyes = sorted(eyes, key=lambda x: x[2]*x[3], reverse=True)
            ex, ey, ew, eh = eyes[0]
            
            eye_roi = gray[ey:ey+eh, ex:ex+ew]
            
            thresh = cv2.adaptiveThreshold(
                eye_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 15, 3
            )
            
            # Remove noise (eyelashes)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 50 < area < 1500: 
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0: continue
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if 0.3 < circularity < 1.2: # Broadened for partial closures
                        valid_contours.append(cnt)

            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M['m00'] != 0:
                    raw_cx = ex + int(M['m10'] / M['m00'])
                    raw_cy = ey + int(M['m01'] / M['m00'])
                    
                    # Exponential Moving Average for smooth output
                    if self.smooth_x is None:
                        self.smooth_x, self.smooth_y = raw_cx, raw_cy
                    else:
                        self.smooth_x = (self.alpha * raw_cx) + ((1 - self.alpha) * self.smooth_x)
                        self.smooth_y = (self.alpha * raw_cy) + ((1 - self.alpha) * self.smooth_y)
                        
                    return (int(self.smooth_x), int(self.smooth_y)), frame

        return None, frame

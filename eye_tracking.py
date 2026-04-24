import cv2
import numpy as np

class EyeTracker:
    def __init__(self):
        # Ensure these XML files are in your GitHub repo!
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    def get_pupil_position(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

            if len(eyes) > 0:
                ex, ey, ew, eh = eyes[0]
                eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                
                # Thresholding to isolate the pupil
                _, thresh = cv2.threshold(eye_img, 40, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        abs_cx = x + ex + cx
                        abs_cy = y + ey + cy
                        
                        return (abs_cx, abs_cy), frame

        return None, frame

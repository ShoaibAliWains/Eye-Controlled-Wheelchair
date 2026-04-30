import cv2
import dlib
import numpy as np
import os
import json
from scipy.spatial import distance as dist
from collections import deque

# ─────────────────────────────────────────────
#  Eye Aspect Ratio  (blink / eye-open check)
# ─────────────────────────────────────────────
def eye_aspect_ratio(eye_points):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Classic Soukupová & Čech formula.
    Returns 0.0 when the eye is fully closed.
    """
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

# ─────────────────────────────────────────────
#  Gaze Ratio  (LEFT / CENTER / RIGHT)
# ─────────────────────────────────────────────
def gaze_ratio(eye_points, gray_frame):
    """
    Divides the eye ROI into left / right halves and
    compares white-pixel count to determine gaze direction.
    """
    height, width = gray_frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(eye_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    eye_region = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    ex, ey = x_coords.min(), y_coords.min()
    ew = x_coords.max() - ex
    eh = y_coords.max() - ey

    if ew <= 0 or eh <= 0:
        return 0.5  # safe default = FORWARD

    eye_crop = eye_region[ey:ey + eh, ex:ex + ew]

    _, thresh = cv2.threshold(eye_crop, 70, 255, cv2.THRESH_BINARY_INV)

    mid = ew // 2
    left_white  = cv2.countNonZero(thresh[:, :mid])
    right_white = cv2.countNonZero(thresh[:, mid:])

    total = left_white + right_white
    if total == 0:
        return 0.5

    ratio = left_white / total  
    return ratio

# ─────────────────────────────────────────────
#  Main EyeTracker class
# ─────────────────────────────────────────────
class EyeTracker:
    LEFT_EYE_IDX  = list(range(36, 42))
    RIGHT_EYE_IDX = list(range(42, 48))
    EAR_CONSEC_FRAMES = 2

    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat", config_path="config.json"):
        # 1. MODEL FILE CHECK
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"\n[CRITICAL ERROR] '{predictor_path}' not found! Please download the Dlib model file into the project folder.\n")

        print("[EyeTracker] Loading dlib face detector …")
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # 2. CONFIG INTEGRATION
        self.ear_threshold = 0.22
        self.gaze_left = 0.65
        self.gaze_right = 0.35
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    self.ear_threshold = cfg.get("ear_blink_threshold", 0.22)
                    self.gaze_left = cfg.get("gaze_left_threshold", 0.65)
                    self.gaze_right = cfg.get("gaze_right_threshold", 0.35)
            except Exception as e:
                print(f"[EyeTracker] Config load error: {e}. Using defaults.")

        self._ratio_buf = deque(maxlen=5)
        self._blink_counter = 0
        self.eye_open = True          
        self.hud_data = None

        print("[EyeTracker] dlib loaded successfully.")

    def _shape_to_np(self, shape):
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def _get_eye_pts(self, landmarks, indices):
        return np.array([(landmarks[i][0], landmarks[i][1]) for i in indices])

    def get_gaze(self, frame):
        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray    = cv2.equalizeHist(gray)   

        faces = self.detector(gray, 0)

        if len(faces) == 0:
            self.eye_open  = False
            self.hud_data  = None
            return "NO_EYE", False, display

        face = max(faces, key=lambda r: r.width() * r.height())
        shape     = self.predictor(gray, face)
        landmarks = self._shape_to_np(shape)

        left_pts  = self._get_eye_pts(landmarks, self.LEFT_EYE_IDX)
        right_pts = self._get_eye_pts(landmarks, self.RIGHT_EYE_IDX)

        ear_l = eye_aspect_ratio(left_pts)
        ear_r = eye_aspect_ratio(right_pts)
        ear   = (ear_l + ear_r) / 2.0

        if ear < self.ear_threshold:
            self._blink_counter += 1
            if self._blink_counter >= self.EAR_CONSEC_FRAMES:
                self.eye_open = False
                self._draw_hud(display, left_pts, right_pts, landmarks,
                               ear, gaze_ratio_val=None, is_closed=True)
                return "NO_EYE", False, display
        else:
            self._blink_counter = 0
            self.eye_open = True

        ratio_l = gaze_ratio(left_pts,  gray)
        ratio_r = gaze_ratio(right_pts, gray)
        raw_ratio = (ratio_l + ratio_r) / 2.0

        self._ratio_buf.append(raw_ratio)
        smooth_ratio = float(np.mean(self._ratio_buf))

        # Dynamic Thresholds
        if smooth_ratio < self.gaze_right:
            gaze_dir = "RIGHT"
        elif smooth_ratio > self.gaze_left:
            gaze_dir = "LEFT"
        else:
            gaze_dir = "FORWARD"

        self._draw_hud(display, left_pts, right_pts, landmarks,
                       ear, smooth_ratio, is_closed=False)

        return gaze_dir, True, display

    def _draw_hud(self, frame, left_pts, right_pts, landmarks, ear, gaze_ratio_val, is_closed):
        color = (0, 80, 255) if is_closed else (0, 255, 80) 

        for pts in [left_pts, right_pts]:
            hull = cv2.convexHull(pts)
            cv2.polylines(frame, [hull], isClosed=True, color=color, thickness=1)

            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())

            if not is_closed:
                cv2.line(frame, (cx - 8, cy),     (cx + 8, cy),     (0, 255, 80), 1)
                cv2.line(frame, (cx,     cy - 8), (cx,     cy + 8), (0, 255, 80), 1)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 80), -1)
            else:
                cv2.line(frame, (cx - 6, cy - 6), (cx + 6, cy + 6), (0, 80, 255), 2)
                cv2.line(frame, (cx + 6, cy - 6), (cx - 6, cy + 6), (0, 80, 255), 2)

        ear_txt = f"EAR:{ear:.2f}"
        cv2.putText(frame, ear_txt,
                    (left_pts[:, 0].min(), left_pts[:, 1].max() + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        if gaze_ratio_val is not None:
            ratio_txt = f"GAZE:{gaze_ratio_val:.2f}"
            cv2.putText(frame, ratio_txt,
                        (right_pts[:, 0].max() - 70, right_pts[:, 1].max() + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

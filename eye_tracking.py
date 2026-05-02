"""
eye_tracking.py  —  Single-Eye Pupil Tracker  (v3  •  glasses-optimised)
=========================================================================

Design goals
------------
* NO dlib, NO 68-point model  →  runs fast on Raspberry Pi Zero / Pi 4
* Works from glasses-mounted camera (close-up, zoomed in, one eye fills frame)
* Gaze direction is RELATIVE to the detected eye box, not to fixed screen %
* Robust to head movement: eye box moves, ratio stays correct
* Graceful fallback: lost eye → last-known ratio for N frames, then NO_EYE

Detection pipeline
------------------
  Frame
   │
   ├─► Haar eye cascade  →  eye bounding box  (ROI)
   │
   ├─► Inside ROI: adaptive threshold  →  isolate dark iris/pupil blob
   │
   ├─► Largest circular contour  →  pupil centre  (cx_local, cy_local)
   │
   ├─► gaze_ratio  =  cx_local / eye_box_width          ← KEY FIX
   │      < left_thr   →  "LEFT"
   │      > right_thr  →  "RIGHT"
   │      else         →  "FORWARD"
   │
   └─► No valid pupil for no_eye_lim frames  →  "NO_EYE"
"""

import cv2
import numpy as np
import json
import os
from collections import deque


# ──────────────────────────────────────────────
#  Default config  (overridden by config.json)
# ──────────────────────────────────────────────
_DEFAULTS = {
    # Gaze ratio thresholds  (0 = left edge of eye box, 1 = right edge)
    "gaze_left_threshold":  0.38,
    "gaze_right_threshold": 0.62,

    # Smoothing: how many frames to average the ratio over
    "ratio_smooth_frames": 6,

    # Pupil contour size filter
    "min_pupil_area": 40,
    "max_pupil_area": 3000,

    # Minimum circularity  (1.0 = perfect circle, lower = allows oval)
    "min_circularity": 0.25,

    # Consecutive frames without pupil before declaring NO_EYE
    "no_eye_frames": 8,

    # Haar cascade tuning  (lower neighbours = more sensitive)
    "haar_scale":      1.15,
    "haar_neighbours": 4,

    # Adaptive threshold params
    "thresh_block": 17,
    "thresh_c":      4,
}


class EyeTracker:
    """
    Single-eye, glasses-mounted, close-up pupil tracker.

    Public API  (identical to v2 so main.py needs NO changes):
        gaze_dir, eye_open, display_frame = tracker.get_gaze(frame)
    """

    def __init__(self, predictor_path=None, config_path="config.json"):
        # predictor_path kept for API compatibility — not used
        cfg = _load_config(config_path)

        self.left_thr   = cfg["gaze_left_threshold"]
        self.right_thr  = cfg["gaze_right_threshold"]
        self.smooth_n   = cfg["ratio_smooth_frames"]
        self.min_area   = cfg["min_pupil_area"]
        self.max_area   = cfg["max_pupil_area"]
        self.min_circ   = cfg["min_circularity"]
        self.no_eye_lim = cfg["no_eye_frames"]
        self.haar_scale = cfg["haar_scale"]
        self.haar_neigh = cfg["haar_neighbours"]
        self.thr_block  = cfg["thresh_block"]
        self.thr_c      = cfg["thresh_c"]

        # Haar cascade — ships with OpenCV, no extra download needed
        cascade_path = "haarcascade_eye.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(
                "[EyeTracker] haarcascade_eye.xml not found — reinstall opencv-python."
            )

        # Internal state
        self._ratio_buf    = deque(maxlen=self.smooth_n)
        self._last_eye_box = None          # stale box for short-miss frames
        self._no_pupil_ctr = 0

        print("[EyeTracker] Loaded — Single-Eye Relative Gaze Tracker (no dlib) ✓")

    # ── public ────────────────────────────────────────────────────────────

    def get_gaze(self, frame):
        """
        Parameters
        ----------
        frame : BGR numpy array from camera

        Returns
        -------
        gaze_dir : "LEFT" | "FORWARD" | "RIGHT" | "NO_EYE"
        eye_open : bool
        display  : BGR frame with HUD drawn on it
        """
        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 1. Detect eye bounding box ────────────────────────────────
        eye_box = self._detect_eye(gray) or self._last_eye_box

        if eye_box is None:
            self._no_pupil_ctr += 1
            return self._no_eye_result(display)

        self._last_eye_box = eye_box
        ex, ey, ew, eh = eye_box

        # Draw eye box (cyan)
        cv2.rectangle(display, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 1)

        # ── 2. Find pupil inside the eye ROI ─────────────────────────
        roi  = gray[ey:ey + eh, ex:ex + ew]
        px, py, contour = self._find_pupil(roi)

        if px is None:
            self._no_pupil_ctr += 1
            if self._no_pupil_ctr >= self.no_eye_lim:
                return self._no_eye_result(display)
            # Short miss — keep last ratio, no command change
            ratio    = float(np.mean(self._ratio_buf)) if self._ratio_buf else 0.5
            gaze_dir = self._ratio_to_dir(ratio)
            return gaze_dir, True, display

        self._no_pupil_ctr = 0

        # ── 3. Gaze ratio  ← THE KEY FIX ─────────────────────────────
        # px is the pupil x-coordinate inside the eye box (local coords).
        # Dividing by box width gives a value in [0, 1] that is INDEPENDENT
        # of where on screen the eye box sits.  Head movement shifts the box
        # but the ratio stays correct.
        ratio = px / max(ew, 1)
        self._ratio_buf.append(ratio)
        smooth = float(np.mean(self._ratio_buf))
        gaze_dir = self._ratio_to_dir(smooth)

        # ── 4. Draw Iron-Man HUD ──────────────────────────────────────
        abs_cx, abs_cy = ex + px, ey + py
        self._draw_hud(display, ex, ey, ew, eh,
                       abs_cx, abs_cy, contour, smooth, gaze_dir)

        return gaze_dir, True, display

    # ── internals ─────────────────────────────────────────────────────────

    def _detect_eye(self, gray):
        """Return largest detected eye box or None."""
        eyes = self._cascade.detectMultiScale(
            gray,
            scaleFactor  = self.haar_scale,
            minNeighbors = self.haar_neigh,
            minSize      = (30, 20),
            maxSize      = (300, 200),
        )
        if len(eyes) == 0:
            return None
        return tuple(sorted(eyes, key=lambda b: b[2] * b[3], reverse=True)[0])

    def _find_pupil(self, roi_gray):
        """
        Find the darkest circular blob (pupil) in the eye ROI.
        Returns (cx, cy, contour) in ROI-local coords, or (None, None, None).
        """
        thresh = cv2.adaptiveThreshold(
            roi_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.thr_block, self.thr_c,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best, best_score = None, -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = 4 * np.pi * area / (perim ** 2)
            if circ < self.min_circ:
                continue
            score = area * circ          # prefer large, round blobs
            if score > best_score:
                best_score, best = score, cnt

        if best is None:
            return None, None, None

        M = cv2.moments(best)
        if M["m00"] == 0:
            return None, None, None

        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), best

    def _ratio_to_dir(self, ratio):
        if ratio < self.left_thr:
            return "LEFT"
        if ratio > self.right_thr:
            return "RIGHT"
        return "FORWARD"

    def _no_eye_result(self, display):
        h, w = display.shape[:2]
        cv2.putText(display, "NO EYE DETECTED",
                    (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2)
        return "NO_EYE", False, display

    def _draw_hud(self, frame, ex, ey, ew, eh,
                  abs_cx, abs_cy, contour, ratio, gaze_dir):

        # Pupil contour outline
        if contour is not None:
            shifted = contour + np.array([[[ex, ey]]])
            cv2.drawContours(frame, [shifted], -1, (0, 255, 80), 1)

        # Pupil crosshair + dot
        cv2.circle(frame, (abs_cx, abs_cy), 4, (0, 255, 80), -1)
        cv2.line(frame, (abs_cx - 10, abs_cy), (abs_cx + 10, abs_cy), (0, 255, 80), 1)
        cv2.line(frame, (abs_cx, abs_cy - 10), (abs_cx, abs_cy + 10), (0, 255, 80), 1)

        # Zone boundaries drawn INSIDE the eye box (not on screen edges)
        lx = int(ex + ew * self.left_thr)
        rx = int(ex + ew * self.right_thr)
        cv2.line(frame, (lx, ey), (lx, ey + eh), (255, 150, 0), 1)
        cv2.line(frame, (rx, ey), (rx, ey + eh), (255, 150, 0), 1)

        # Gaze ratio bar (just below eye box)
        bx, by, bw, bh = ex, ey + eh + 6, ew, 6
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        fill   = int(bw * ratio)
        bcolor = (0, 255, 80) if gaze_dir == "FORWARD" else (0, 120, 255)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), bcolor, -1)
        cv2.putText(frame, f"{ratio:.2f}",
                    (bx + bw + 6, by + bh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


# ── module-level helper ───────────────────────────────────────────────────────

def _load_config(path):
    cfg = _DEFAULTS.copy()
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update(json.load(f))
            print(f"[EyeTracker] Config loaded: {path}")
        except Exception as e:
            print(f"[EyeTracker] Config error ({e}) — using defaults")
    return cfg

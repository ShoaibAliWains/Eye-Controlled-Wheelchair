"""
eye_tracking.py  —  Single-Eye Pupil Tracker
=============================================
Built on the working v1 base (Haar cascade + contours).

KEY UPGRADES
------------
1. Gaze RATIO instead of absolute pixel position
   - ratio = pupil_x_inside_eye_box / eye_box_width
   - Head movement shifts the box AND the pupil together → ratio stays stable
   - No blue square needed, no calibration needed

2. Dynamic pupil area (scales with eye-box size)
   - Works close-up (glasses) AND mid-range camera

3. Eyelash rejection
   - Circularity filter: eyelashes are thin strips (circ ~0.05), iris is round (circ ~0.6+)
   - Morphological CLOSE fills the iris blob so it beats eyelashes in area

4. Temporal validation of eye box
   - Must appear in 3 consecutive frames before accepted
   - Kills single-frame Haar jumps

5. EMA smoothing on gaze ratio (like v1 had on x/y)

Returns:  gaze_dir (str), eye_open (bool), display_frame (BGR)
"""

import cv2
import numpy as np
import json
import os
from collections import deque


# ── tuneable defaults ─────────────────────────────────────────────────────────
_DEFAULTS = {
    "gaze_left_threshold":  0.42,   # ratio < this → LEFT
    "gaze_right_threshold": 0.58,   # ratio > this → RIGHT
    "ratio_smooth_frames":  7,      # EMA window
    "pupil_min_fraction":   0.04,   # min pupil area as % of eye-box area
    "pupil_max_fraction":   0.40,   # max pupil area as % of eye-box area
    "min_circularity":      0.35,   # rejects eyelashes (they score ~0.05-0.15)
    "darkness_percentile":  40,     # threshold on darkest N% of ROI
    "no_eye_frames":        10,     # consecutive misses before NO_EYE
    "haar_scale":           1.2,
    "haar_neighbours":      4,
    "eye_box_confirm":      3,      # frames before eye box is accepted
}


def _load_cfg(path="config.json"):
    cfg = _DEFAULTS.copy()
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[EyeTracker] config error: {e}")
    return cfg


class EyeTracker:
    def __init__(self, config_path="config.json"):
        cfg = _load_cfg(config_path)

        self.left_thr  = cfg["gaze_left_threshold"]
        self.right_thr = cfg["gaze_right_threshold"]
        self._min_frac = cfg["pupil_min_fraction"]
        self._max_frac = cfg["pupil_max_fraction"]
        self._min_circ = cfg["min_circularity"]
        self._dark_pct = cfg["darkness_percentile"]
        self._no_lim   = cfg["no_eye_frames"]
        self._confirm  = cfg["eye_box_confirm"]
        self._smooth_n = cfg["ratio_smooth_frames"]

        # Haar cascade — same as v1 but with correct Pi path
        self._cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        if self._cascade.empty():
            raise RuntimeError(
                "[EyeTracker] haarcascade_eye.xml not found!\n"
                "Run: find / -name haarcascade_eye.xml 2>/dev/null\n"
                "Then: cp <found_path> ~/Eye-controlled-wheelchair/"
            )

        # State
        self._ratio_buf   = deque(maxlen=self._smooth_n)
        self._box_history = deque(maxlen=self._confirm)
        self._last_box    = None
        self._no_ctr      = 0

        # Public flags (main.py reads these)
        self.calibrating  = False   # always False — no calibration needed

        print("[EyeTracker] Ready — relative gaze ratio mode (no calibration) ✓")
        print(f"[EyeTracker] LEFT < {self.left_thr}  |  RIGHT > {self.right_thr}")

    # ── public API ────────────────────────────────────────────────────────────

    def get_gaze(self, frame):
        """
        Returns (gaze_dir, eye_open, display_frame)
        gaze_dir : "LEFT" | "FORWARD" | "RIGHT" | "NO_EYE"
        eye_open : bool
        """
        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray    = cv2.equalizeHist(gray)   # normalise brightness / contrast

        # 1. Detect eye box (temporal validation)
        box = self._detect_eye(gray) or self._last_box

        if box is None:
            self._no_ctr += 1
            return self._no_eye(display)

        self._last_box = box
        ex, ey, ew, eh = box

        # Draw eye bounding box (cyan)
        cv2.rectangle(display, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 1)

        # 2. Find pupil inside eye ROI
        roi = gray[ey:ey + eh, ex:ex + ew]
        px, py, contour = self._find_pupil(roi, ew, eh)

        if px is None:
            self._no_ctr += 1
            if self._no_ctr >= self._no_lim:
                return self._no_eye(display)
            # Short miss → keep last ratio
            ratio    = float(np.mean(self._ratio_buf)) if self._ratio_buf else 0.5
            gaze_dir = self._dir(ratio)
            return gaze_dir, True, display

        self._no_ctr = 0

        # 3. KEY FIX: gaze ratio = pupil position INSIDE the eye box
        ratio = px / max(ew, 1)           # 0=left edge, 1=right edge of eye box
        self._ratio_buf.append(ratio)
        smooth   = float(np.mean(self._ratio_buf))
        gaze_dir = self._dir(smooth)

        # 4. Draw HUD
        abs_cx, abs_cy = ex + px, ey + py
        self._hud(display, ex, ey, ew, eh, abs_cx, abs_cy,
                  contour, smooth, gaze_dir)

        return gaze_dir, True, display

    # ── internal helpers ──────────────────────────────────────────────────────

    def _detect_eye(self, gray):
        """Haar detect → temporal validation → median-smoothed box."""
        raw = self._cascade.detectMultiScale(
            gray,
            scaleFactor  = 1.2,
            minNeighbors = 4,
            minSize      = (30, 20),
            maxSize      = (600, 400),   # large for close-up glasses camera
        )

        best = None
        if len(raw) > 0:
            # Real eyes are wider than tall
            valid = [b for b in raw if b[2] / max(b[3], 1) > 1.1]
            if valid:
                best = tuple(sorted(valid,
                                    key=lambda b: b[2] * b[3],
                                    reverse=True)[0])

        self._box_history.append(best)

        if (len(self._box_history) == self._confirm and
                all(b is not None for b in self._box_history)):
            xs = [b[0] for b in self._box_history]
            ys = [b[1] for b in self._box_history]
            ws = [b[2] for b in self._box_history]
            hs = [b[3] for b in self._box_history]
            return (int(np.median(xs)), int(np.median(ys)),
                    int(np.median(ws)), int(np.median(hs)))
        return None

    def _find_pupil(self, roi_gray, ew, eh):
        """
        Find darkest circular blob (pupil) in eye ROI.
        Eyelashes rejected by circularity < min_circularity.
        """
        box_area = max(ew * eh, 1)
        min_a    = box_area * self._min_frac
        max_a    = box_area * self._max_frac

        # Only look at the darkest pixels
        dark_level = int(np.percentile(roi_gray, self._dark_pct))
        _, thresh  = cv2.threshold(roi_gray, dark_level, 255,
                                   cv2.THRESH_BINARY_INV)

        # CLOSE fills iris blob; OPEN removes eyelash noise
        ell5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ell3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ell5, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  ell3, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        best, best_score = None, -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_a < area < max_a):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = 4 * np.pi * area / (perim ** 2)
            if circ < self._min_circ:           # ← EYELASH FILTER
                continue
            score = area * circ                 # large + round wins
            if score > best_score:
                best_score, best = score, cnt

        if best is None:
            return None, None, None

        M = cv2.moments(best)
        if M["m00"] == 0:
            return None, None, None

        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), best

    def _dir(self, ratio):
        if ratio < self.left_thr:
            return "LEFT"
        if ratio > self.right_thr:
            return "RIGHT"
        return "FORWARD"

    def _no_eye(self, display):
        h, w = display.shape[:2]
        cv2.putText(display, "NO EYE DETECTED",
                    (w // 2 - 130, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2)
        return "NO_EYE", False, display

    def _hud(self, frame, ex, ey, ew, eh,
             abs_cx, abs_cy, contour, ratio, gaze_dir):

        # Pupil contour + crosshair
        if contour is not None:
            shifted = contour + np.array([[[ex, ey]]])
            cv2.drawContours(frame, [shifted], -1, (0, 255, 80), 1)
        cv2.circle(frame, (abs_cx, abs_cy), 5, (0, 255, 80), -1)
        cv2.line(frame, (abs_cx - 12, abs_cy), (abs_cx + 12, abs_cy), (0, 255, 80), 1)
        cv2.line(frame, (abs_cx, abs_cy - 12), (abs_cx, abs_cy + 12), (0, 255, 80), 1)

        # Zone lines INSIDE the eye box (not screen edges)
        lx = int(ex + ew * self.left_thr)
        rx = int(ex + ew * self.right_thr)
        cv2.line(frame, (lx, ey), (lx, ey + eh), (255, 150, 0), 1)
        cv2.line(frame, (rx, ey), (rx, ey + eh), (255, 150, 0), 1)

        # Ratio bar below eye box
        bx, by, bw, bh = ex, ey + eh + 5, ew, 7
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
        fill = int(bw * float(np.clip(ratio, 0, 1)))
        col  = (0, 255, 80) if gaze_dir == "FORWARD" else (0, 100, 255)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), col, -1)
        cv2.putText(frame, f"{ratio:.2f}  {gaze_dir}",
                    (bx, by - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

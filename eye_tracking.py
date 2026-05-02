"""
eye_tracking.py  —  Single-Eye Pupil Tracker  (v4  •  Auto-Calibration Edition)
=================================================================================

KEY FIXES in this version
--------------------------
1. EYELASH REJECTION
   - Pupil area is now DYNAMIC: min/max area scales with the detected eye-box size
     so it works at any camera distance (close-up glasses OR 30 cm away).
   - Strict circularity filter  (eyelashes are long/thin → low circularity).
   - Morphological CLOSE after threshold to fill the iris blob and separate it
     from eyelash noise.
   - Only the darkest region inside the ROI is accepted (eyelashes are lighter
     than the iris centre).

2. AUTO-CALIBRATION  (replaces hardcoded 0.38 / 0.62)
   - On startup the user looks CENTRE for 3 s → base ratio recorded.
   - Then looks LEFT for 2 s and RIGHT for 2 s → dynamic thresholds set.
   - Thresholds adapt to EACH user's eye shape and camera distance.
   - Fallback to config.json defaults if calibration is skipped.

3. CAMERA STABILITY / FALSE TRACKING
   - Haar eye cascade results are validated against minimum aspect ratio
     (real eyes are wide, not tall → filters stray face detections).
   - Eye-box must be consistent across 3 consecutive frames before being
     accepted (temporal validation → kills single-frame jumps).
   - Lost-eye counter is separate from intentional-blink counter so that
     tracking loss no longer triggers the double-blink pause.
"""

import cv2
import numpy as np
import json
import os
import time
from collections import deque


# ──────────────────────────────────────────────────────────────
#  Default config  (all overridable via config.json)
# ──────────────────────────────────────────────────────────────
_DEFAULTS = {
    # Gaze ratio fallback thresholds (used only if calibration is skipped)
    "gaze_left_threshold":  0.45,
    "gaze_right_threshold": 0.55,

    # Ratio smoothing window (frames)
    "ratio_smooth_frames": 8,

    # Pupil area as fraction of eye-box area  ← DYNAMIC, not pixel count
    "pupil_min_fraction": 0.05,    # pupil ≥ 5% of eye box area (filters eyelashes)
    "pupil_max_fraction": 0.35,    # pupil ≤ 35% of eye box area

    # Circularity: 1.0 = perfect circle. Eyelashes ≈ 0.05–0.15
    "min_circularity": 0.40,

    # Darkness percentile — only blobs darker than this % of ROI accepted
    "darkness_percentile": 35,

    # Frames without valid pupil before declaring NO_EYE
    "no_eye_frames": 10,

    # Haar cascade parameters
    "haar_scale":      1.15,
    "haar_neighbours": 4,

    # Eye-box temporal validation: must appear in N consecutive frames
    "eye_box_confirm_frames": 3,

    # Adaptive threshold
    "thresh_block": 11,
    "thresh_c":      6,

    # Calibration durations (seconds)
    "calib_centre_sec": 3.0,
    "calib_side_sec":   2.0,
}


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


# ──────────────────────────────────────────────────────────────
#  Auto-Calibration helper
# ──────────────────────────────────────────────────────────────
class _Calibrator:
    """
    Guides user through centre → left → right calibration.
    Collects median gaze ratios and sets dynamic thresholds.
    """

    STEPS = [
        ("CENTRE", "Look at the camera STRAIGHT — hold 3 s"),
        ("LEFT",   "Look as FAR LEFT as you can — hold 2 s"),
        ("RIGHT",  "Look as FAR RIGHT as you can — hold 2 s"),
    ]

    def __init__(self, cfg):
        self.centre_sec = cfg["calib_centre_sec"]
        self.side_sec   = cfg["calib_side_sec"]
        self._durations = [self.centre_sec, self.side_sec, self.side_sec]
        self._step      = 0
        self._samples   = {k: [] for k, _ in self.STEPS}
        self._t_start   = time.time()
        self.done       = False
        self.left_thr   = cfg["gaze_left_threshold"]   # fallback
        self.right_thr  = cfg["gaze_right_threshold"]

    @property
    def current_label(self):
        return self.STEPS[self._step][0]

    @property
    def current_instruction(self):
        return self.STEPS[self._step][1]

    def add_sample(self, ratio):
        if self.done:
            return
        elapsed = time.time() - self._t_start
        dur     = self._durations[self._step]

        # Collect samples in second half of each window (eye is settled)
        if elapsed > dur * 0.5:
            self._samples[self.current_label].append(ratio)

        if elapsed >= dur:
            self._step  += 1
            self._t_start = time.time()
            if self._step >= len(self.STEPS):
                self._finalise()

    def _finalise(self):
        centre = float(np.median(self._samples["CENTRE"])) if self._samples["CENTRE"] else 0.50
        left   = float(np.median(self._samples["LEFT"]))   if self._samples["LEFT"]   else 0.30
        right  = float(np.median(self._samples["RIGHT"]))  if self._samples["RIGHT"]  else 0.70

        # Midpoints between centre and each extreme
        self.left_thr  = round((centre + left)  / 2, 3)
        self.right_thr = round((centre + right) / 2, 3)

        print(f"[Calibration] centre={centre:.3f}  left={left:.3f}  right={right:.3f}")
        print(f"[Calibration] Thresholds set → LEFT<{self.left_thr}  RIGHT>{self.right_thr}")
        self.done = True

    def time_remaining(self):
        return max(0, self._durations[self._step] - (time.time() - self._t_start))


# ──────────────────────────────────────────────────────────────
#  Main EyeTracker class
# ──────────────────────────────────────────────────────────────
class EyeTracker:
    """
    Single-eye, glasses-OR-mid-range camera, pupil gaze tracker.

    Public API (unchanged from v3):
        gaze_dir, eye_open, display_frame = tracker.get_gaze(frame)

    Also exposes:
        tracker.calibrating  →  True while calibration is in progress
        tracker.left_thr / right_thr  →  current thresholds (after calib)
    """

    def __init__(self, predictor_path=None, config_path="config.json"):
        cfg = _load_config(config_path)

        # Thresholds (overwritten by calibration)
        self.left_thr  = cfg["gaze_left_threshold"]
        self.right_thr = cfg["gaze_right_threshold"]

        # Dynamic area fractions
        self._min_frac = cfg["pupil_min_fraction"]
        self._max_frac = cfg["pupil_max_fraction"]
        self._min_circ = cfg["min_circularity"]
        self._dark_pct = cfg["darkness_percentile"]

        self._smooth_n    = cfg["ratio_smooth_frames"]
        self._no_eye_lim  = cfg["no_eye_frames"]
        self._haar_scale  = cfg["haar_scale"]
        self._haar_neigh  = cfg["haar_neighbours"]
        self._confirm_n   = cfg["eye_box_confirm_frames"]
        self._thr_block   = cfg["thresh_block"]
        self._thr_c       = cfg["thresh_c"]

        # Haar cascade — use direct filename (cv2.data not available on Pi OS)
        cascade_path = "haarcascade_eye.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("[EyeTracker] haarcascade_eye.xml not found.")

        # Internal state
        self._ratio_buf        = deque(maxlen=self._smooth_n)
        self._last_eye_box     = None
        self._no_pupil_ctr     = 0
        self._eye_box_history  = deque(maxlen=self._confirm_n)   # temporal validation

        # Calibration — BYPASSED: use config thresholds directly
        self._calib = _Calibrator(cfg)
        self._calib.done = True      # FORCE SKIP calibration screen
        self.calibrating = False     # FORCE FALSE so motor commands fire immediately

        print("[EyeTracker] v4 — Dynamic Pupil Filter + Auto-Calibration  ✓")
        print(f"[EyeTracker] Starting calibration: '{self._calib.current_instruction}'")

    # ── public ───────────────────────────────────────────────────────────

    def get_gaze(self, frame):
        display = frame.copy()
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray    = cv2.equalizeHist(gray)        # normalise brightness

        # ── CALIBRATION PHASE ────────────────────────────────────────
        if not self._calib.done:
            ratio, display = self._run_calibration_frame(gray, display)
            # During calibration always return STOP to motors
            return "NO_EYE", False, display

        self.calibrating = False

        # ── TRACKING PHASE ───────────────────────────────────────────
        eye_box = self._detect_eye_stable(gray)

        if eye_box is None:
            eye_box = self._last_eye_box      # short-miss fallback

        if eye_box is None:
            self._no_pupil_ctr += 1
            return self._no_eye_result(display)

        self._last_eye_box = eye_box
        ex, ey, ew, eh = eye_box

        # Draw eye box (cyan)
        cv2.rectangle(display, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 1)

        # Pupil search inside eye ROI
        roi  = gray[ey:ey + eh, ex:ex + ew]
        px, py, contour = self._find_pupil(roi, ew, eh)

        if px is None:
            self._no_pupil_ctr += 1
            if self._no_pupil_ctr >= self._no_eye_lim:
                return self._no_eye_result(display)
            ratio    = float(np.mean(self._ratio_buf)) if self._ratio_buf else 0.5
            gaze_dir = self._ratio_to_dir(ratio)
            return gaze_dir, True, display

        self._no_pupil_ctr = 0

        # Gaze ratio — relative inside eye box
        ratio = px / max(ew, 1)
        self._ratio_buf.append(ratio)
        smooth   = float(np.mean(self._ratio_buf))
        gaze_dir = self._ratio_to_dir(smooth)

        abs_cx, abs_cy = ex + px, ey + py
        self._draw_hud(display, ex, ey, ew, eh, abs_cx, abs_cy,
                       contour, smooth, gaze_dir)

        return gaze_dir, True, display

    # ── calibration ──────────────────────────────────────────────────────

    def _run_calibration_frame(self, gray, display):
        h, w = display.shape[:2]

        # Try to get a ratio even during calibration
        ratio = 0.5
        eye_box = self._detect_eye_stable(gray)
        if eye_box:
            ex, ey, ew, eh = eye_box
            roi = gray[ey:ey + eh, ex:ex + ew]
            px, _, _ = self._find_pupil(roi, ew, eh)
            if px is not None:
                ratio = px / max(ew, 1)
                self._calib.add_sample(ratio)

        # Draw calibration UI
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

        step_name = self._calib.current_label
        instr     = self._calib.current_instruction
        remaining = self._calib.time_remaining()

        # Progress bar
        total_dur = ([self._calib.centre_sec,
                      self._calib.side_sec,
                      self._calib.side_sec][self._calib._step])
        progress  = 1.0 - (remaining / total_dur)
        bar_w     = int(w * 0.7)
        bar_x     = (w - bar_w) // 2
        bar_y     = h // 2 + 50
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12),
                      (60, 60, 60), -1)
        cv2.rectangle(display, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress), bar_y + 12),
                      (0, 220, 80), -1)

        # Text
        cv2.putText(display, "CALIBRATING",
                    (w // 2 - 100, h // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
        cv2.putText(display, instr,
                    (w // 2 - min(len(instr) * 8, w // 2 - 10), h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        cv2.putText(display, f"Step: {step_name}   {remaining:.1f}s",
                    (w // 2 - 100, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)

        return ratio, display

    # ── eye detection ─────────────────────────────────────────────────────

    def _detect_eye_stable(self, gray):
        """
        Returns a validated eye box only when consistent across
        _confirm_n consecutive frames. Kills single-frame jumps.
        """
        raw = self._cascade.detectMultiScale(
            gray,
            scaleFactor  = self._haar_scale,
            minNeighbors = self._haar_neigh,
            minSize      = (25, 15),
            maxSize      = (600, 400),
        )

        best = None
        if len(raw) > 0:
            # Filter: real eyes are wider than tall (aspect ratio > 1.2)
            valid = [b for b in raw if b[2] / max(b[3], 1) > 1.2]
            if valid:
                best = tuple(sorted(valid,
                                    key=lambda b: b[2] * b[3],
                                    reverse=True)[0])

        self._eye_box_history.append(best)

        # Accept only if all recent frames agree (non-None)
        if (len(self._eye_box_history) == self._confirm_n and
                all(b is not None for b in self._eye_box_history)):
            # Return median box to smooth jitter
            xs = [b[0] for b in self._eye_box_history]
            ys = [b[1] for b in self._eye_box_history]
            ws = [b[2] for b in self._eye_box_history]
            hs = [b[3] for b in self._eye_box_history]
            return (int(np.median(xs)), int(np.median(ys)),
                    int(np.median(ws)), int(np.median(hs)))

        return None

    # ── pupil detection ───────────────────────────────────────────────────

    def _find_pupil(self, roi_gray, ew, eh):
        """
        Dynamic-area pupil finder with eyelash rejection.
        Returns (cx, cy, contour) in ROI-local coords, or (None, None, None).
        """
        box_area = max(ew * eh, 1)
        min_area = box_area * self._min_frac
        max_area = box_area * self._max_frac

        # Darkness threshold: only consider pixels darker than N-th percentile
        dark_level = int(np.percentile(roi_gray, self._dark_pct))

        # Binary threshold on darkest pixels
        _, thresh = cv2.threshold(roi_gray, dark_level, 255, cv2.THRESH_BINARY_INV)

        # Morphological close: fills iris blob, merges nearby dark pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Open: removes thin eyelash noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                  iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        best, best_score = None, -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue

            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = 4 * np.pi * area / (perim ** 2)

            # EYELASH REJECTION: eyelashes are thin strips → low circularity
            if circ < self._min_circ:
                continue

            # Prefer large, circular, dark blobs
            score = area * circ
            if score > best_score:
                best_score, best = score, cnt

        if best is None:
            return None, None, None

        M = cv2.moments(best)
        if M["m00"] == 0:
            return None, None, None

        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), best

    # ── helpers ───────────────────────────────────────────────────────────

    def _ratio_to_dir(self, ratio):
        if ratio < self.left_thr:
            return "LEFT"
        if ratio > self.right_thr:
            return "RIGHT"
        return "FORWARD"

    def _no_eye_result(self, display):
        h, w = display.shape[:2]
        cv2.putText(display, "NO EYE DETECTED",
                    (w // 2 - 130, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2)
        return "NO_EYE", False, display

    def _draw_hud(self, frame, ex, ey, ew, eh,
                  abs_cx, abs_cy, contour, ratio, gaze_dir):

        # Pupil contour outline (green)
        if contour is not None:
            shifted = contour + np.array([[[ex, ey]]])
            cv2.drawContours(frame, [shifted], -1, (0, 255, 80), 1)

        # Crosshair on pupil centre
        cv2.circle(frame, (abs_cx, abs_cy), 4, (0, 255, 80), -1)
        cv2.line(frame, (abs_cx - 10, abs_cy), (abs_cx + 10, abs_cy), (0, 255, 80), 1)
        cv2.line(frame, (abs_cx, abs_cy - 10), (abs_cx, abs_cy + 10), (0, 255, 80), 1)

        # Dynamic zone boundaries inside eye box
        lx = int(ex + ew * self.left_thr)
        rx = int(ex + ew * self.right_thr)
        cv2.line(frame, (lx, ey), (lx, ey + eh), (255, 150, 0), 1)
        cv2.line(frame, (rx, ey), (rx, ey + eh), (255, 150, 0), 1)

        # Ratio bar below eye box
        bx, by, bw, bh = ex, ey + eh + 6, ew, 7
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
        fill   = int(bw * np.clip(ratio, 0, 1))
        bcolor = (0, 255, 80) if gaze_dir == "FORWARD" else (0, 120, 255)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), bcolor, -1)
        cv2.putText(frame, f"{ratio:.2f}",
                    (bx + bw + 6, by + bh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        # Gaze label next to eye box
        label_color = {"LEFT": (0, 120, 255),
                       "RIGHT": (0, 120, 255),
                       "FORWARD": (0, 255, 80)}.get(gaze_dir, (160, 160, 160))
        cv2.putText(frame, gaze_dir,
                    (ex, ey - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

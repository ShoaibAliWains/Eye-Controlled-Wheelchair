"""
control_logic.py  —  Gaze-to-Command Translator  (v4)
======================================================

KEY FIXES in this version
--------------------------
1. TRACKING LOSS ≠ BLINK
   - "NO_EYE" from the tracker (Haar failed to find eye box) is treated as
     TRACKING LOSS, not an intentional blink.
   - Only EAR-confirmed short blinks (eye_open=False while eye IS visible)
     are counted toward the double-blink pause trigger.
   - This prevents random Haar mis-detections from pausing the wheelchair.

2. DOUBLE-BLINK COOLDOWN
   - After a double-blink fires, a strict 2-second cooldown prevents
     immediate re-trigger (eliminates rapid PAUSE ↔ READY toggling).

3. CALIBRATION AWARENESS
   - If EyeTracker is still calibrating, LogicController returns "STOP"
     without touching the blink or debounce state.
"""

import time
import json
import os
from collections import deque


DEFAULT_CONFIG = {
    "hold_frames":          10,     # stable frames before command fires
    "blink_tolerance":       0.5,   # seconds tracking loss tolerated before E-STOP
    "double_blink_win":      1.2,   # window (s) for counting intentional blinks
    "double_blink_cooldown": 2.0,   # cooldown (s) after a double-blink fires
    "paused_on_start":      False,
}


def load_config(path="config.json"):
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[Logic] Config error ({e}) — using defaults")
    return cfg


class LogicController:

    def __init__(self, config_path="config.json"):
        cfg = load_config(config_path)

        self.state = "READY"

        # Debounce
        self.hold_frames     = cfg["hold_frames"]
        self._history        = deque(maxlen=self.hold_frames)
        self._last_confirmed = "STOP"

        # Tracking-loss E-STOP
        self.blink_tolerance  = cfg["blink_tolerance"]
        self._last_eye_time   = time.time()

        # Double-blink PAUSE  (only counts intentional blinks, not tracking loss)
        self._dbl_win      = cfg["double_blink_win"]
        self._dbl_cooldown = cfg["double_blink_cooldown"]
        self._blink_times  = deque(maxlen=10)
        self._last_dbl_t   = 0.0          # timestamp of last double-blink fire
        self._paused       = cfg["paused_on_start"]

        self.current_command = "STOP"

    # ── public API ────────────────────────────────────────────────────────

    def process(self, gaze_dir: str, eye_open: bool,
                calibrating: bool = False) -> str:
        """
        Parameters
        ----------
        gaze_dir    : "LEFT" | "FORWARD" | "RIGHT" | "NO_EYE"
        eye_open    : False = blink detected BY TRACKER (eye visible but closed)
        calibrating : True while EyeTracker is still in calibration phase

        Returns
        -------
        command : one of "FORWARD" | "LEFT" | "RIGHT" | "STOP" |
                         "EMERGENCY STOP - NO EYE" | "PAUSED" | "CALIBRATING"
        """
        # ── 0. Calibration block ──────────────────────────────────────
        if calibrating:
            self.current_command = "CALIBRATING..."
            self._last_eye_time  = time.time()   # reset so no false E-STOP
            return "CALIBRATING..."

        now = time.time()

        # ── 1. Tracking loss (Haar eye box not found) ─────────────────
        if gaze_dir == "NO_EYE":
            elapsed = now - self._last_eye_time
            # Do NOT count this as an intentional blink
            if elapsed > self.blink_tolerance:
                self._history.clear()
                self.current_command = "EMERGENCY STOP - NO EYE"
                return self.current_command
            else:
                # Short loss — hold last known command, no jerk
                return self._last_confirmed

        # Eye is visible from here on
        self._last_eye_time = now

        # ── 2. Intentional blink (eye visible but closed) → double-blink
        if not eye_open:
            self._blink_times.append(now)
            if self._check_double_blink(now):
                self._paused = not self._paused
                action = "PAUSED" if self._paused else "READY"
                print(f"[Logic] Double-blink → {action}")
            # During a blink, hold last command
            return self._last_confirmed

        # ── 3. Paused ─────────────────────────────────────────────────
        if self._paused:
            self.current_command = "PAUSED"
            return "PAUSED"

        # ── 4. Debounce ───────────────────────────────────────────────
        self._history.append(gaze_dir)

        if (len(self._history) == self.hold_frames and
                len(set(self._history)) == 1):
            confirmed = self._history[-1]
            self._last_confirmed = confirmed
            self.current_command = confirmed
            return confirmed

        self.current_command = "HOLDING..."
        return "HOLDING..."

    def reset(self):
        self.state           = "READY"
        self._history.clear()
        self._last_confirmed = "STOP"
        self._paused         = False
        self._last_eye_time  = time.time()
        self._blink_times.clear()
        print("[Logic] Reset.")

    # ── internal ──────────────────────────────────────────────────────────

    def _check_double_blink(self, now):
        # Enforce cooldown — prevents rapid re-trigger
        if now - self._last_dbl_t < self._dbl_cooldown:
            return False

        recent = [t for t in self._blink_times if now - t < self._dbl_win]
        if len(recent) >= 2:
            self._blink_times.clear()
            self._last_dbl_t = now
            return True
        return False

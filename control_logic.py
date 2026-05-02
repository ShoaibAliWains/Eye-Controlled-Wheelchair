"""
control_logic.py  —  Gaze → Motor Command
==========================================
Upgraded from v1 base.

KEY CHANGES vs v1
-----------------
• Input is now gaze_dir STRING ("LEFT"/"FORWARD"/"RIGHT"/"NO_EYE")
  instead of raw pixel (px, py) — no more center_x/center_y math here
• Tracking loss (NO_EYE) and intentional blink are separated
  → tracking loss never triggers double-blink pause
• Double-blink pause has 2-second cooldown to prevent rapid toggling
• Debounce works same way as v1 but on string history
"""

import time
import json
import os
from collections import deque


_DEFAULTS = {
    "hold_frames":          12,     # frames gaze must be stable (≈0.4s at 30fps)
    "blink_tolerance":       0.5,   # seconds tracking loss before E-STOP
    "double_blink_win":      1.2,   # window for double-blink detection (s)
    "double_blink_cooldown": 2.0,   # cooldown after double-blink fires (s)
    "paused_on_start":      False,
}


def _load_cfg(path="config.json"):
    cfg = _DEFAULTS.copy()
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[Logic] config error: {e}")
    return cfg


class LogicController:

    def __init__(self, config_path="config.json"):
        cfg = _load_cfg(config_path)

        # Keep state attr so main.py draw_ui works unchanged
        self.state = "READY"

        self._hold    = cfg["hold_frames"]
        self._history = deque(maxlen=self._hold)
        self._last_ok = "STOP"

        self._blink_tol  = cfg["blink_tolerance"]
        self._last_eye_t = time.time()

        self._dbl_win  = cfg["double_blink_win"]
        self._dbl_cool = cfg["double_blink_cooldown"]
        self._blinks   = deque(maxlen=10)
        self._last_dbl = 0.0
        self._paused   = cfg["paused_on_start"]

        self.current_command = "STOP"

    # ── public ────────────────────────────────────────────────────────────────

    def get_command(self, gaze_dir: str, eye_open: bool = True) -> str:
        """
        Parameters
        ----------
        gaze_dir : "LEFT" | "FORWARD" | "RIGHT" | "NO_EYE"
        eye_open : False = intentional blink (eye visible but closed)

        Returns
        -------
        str command for motor controller
        """
        now = time.time()

        # ── 1. Tracking loss (eye box not found at all) ────────────────
        if gaze_dir == "NO_EYE":
            elapsed = now - self._last_eye_t
            # Not an intentional blink — don't count toward double-blink
            if elapsed > self._blink_tol:
                self._history.clear()
                self.current_command = "EMERGENCY STOP - NO EYE"
                return self.current_command
            return self._last_ok   # short miss → hold last command

        self._last_eye_t = now

        # ── 2. Intentional blink (eye visible but closed) ─────────────
        if not eye_open:
            self._blinks.append(now)
            if self._double_blink(now):
                self._paused = not self._paused
                print(f"[Logic] Double-blink → {'PAUSED' if self._paused else 'READY'}")
            return self._last_ok   # hold during blink

        # ── 3. Paused ─────────────────────────────────────────────────
        if self._paused:
            self.current_command = "PAUSED"
            return "PAUSED"

        # ── 4. Debounce — same logic as v1 ────────────────────────────
        self._history.append(gaze_dir)

        if len(self._history) == self._hold and len(set(self._history)) == 1:
            confirmed = self._history[-1]
            self._last_ok        = confirmed
            self.current_command = confirmed
            return confirmed

        # Prioritize STOP if it appeared recently (v1 behaviour kept)
        if "STOP" in list(self._history)[-4:]:
            self.current_command = "STOP"
            return "STOP"

        self.current_command = "HOLDING..."
        return "HOLDING..."

    def reset(self):
        self.state    = "READY"
        self._history.clear()
        self._last_ok = "STOP"
        self._paused  = False
        self._last_eye_t = time.time()
        self._blinks.clear()
        print("[Logic] Reset.")

    # ── internal ──────────────────────────────────────────────────────────────

    def _double_blink(self, now):
        if now - self._last_dbl < self._dbl_cool:
            return False
        recent = [t for t in self._blinks if now - t < self._dbl_win]
        if len(recent) >= 2:
            self._blinks.clear()
            self._last_dbl = now
            return True
        return False

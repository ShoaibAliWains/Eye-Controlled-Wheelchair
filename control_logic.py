import time
import json
import os
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
#  LogicController  —  translates gaze direction into motor commands
#
#  Key improvements over v1:
#   • Works with gaze_dir strings ("LEFT" / "FORWARD" / "RIGHT" / "NO_EYE")
#     instead of raw pixel coordinates → no blue-square dependency
#   • EAR-based eye-open check done in EyeTracker; here we just receive a bool
#   • Debounce: command must be held for `hold_frames` consecutive frames
#   • Double-blink detection: two blinks within 1 s → PAUSE toggle
#   • Blink tolerance: short blinks (<0.6 s) do not trigger E-STOP
#   • Optional JSON config file for easy threshold tuning without editing code
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "hold_frames":      10,      # frames gaze must be stable before command fires
    "blink_tolerance":   0.6,    # seconds of missing eye before E-STOP
    "double_blink_win":  1.0,    # seconds window for double-blink detection
    "paused_on_start":  False    # start in PAUSED state for safety
}


def load_config(path="config.json"):
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(path):
        try:
            with open(path) as f:
                overrides = json.load(f)
            cfg.update(overrides)
            print(f"[Config] Loaded '{path}': {overrides}")
        except Exception as e:
            print(f"[Config] Could not read '{path}': {e} — using defaults.")
    return cfg


class LogicController:

    def __init__(self, config_path="config.json"):
        cfg = load_config(config_path)

        self.state = "READY"   # No calibration needed — gaze ratio is self-relative

        # Debounce
        self.hold_frames     = cfg["hold_frames"]
        self._history        = deque(maxlen=self.hold_frames)
        self._last_confirmed = "STOP"

        # Blink / E-STOP
        self.blink_tolerance = cfg["blink_tolerance"]
        self._last_eye_time  = time.time()

        # Double-blink pause
        self.double_blink_win = cfg["double_blink_win"]
        self._blink_times     = deque(maxlen=5)
        self._paused          = cfg["paused_on_start"]

        # Expose for draw_ui
        self.current_command  = "STOP"

    # ── public API ────────────────────────────────────────────────────────

    def process(self, gaze_dir: str, eye_open: bool) -> str:
        """
        Call once per frame.

        Parameters
        ----------
        gaze_dir : "LEFT" | "FORWARD" | "RIGHT" | "NO_EYE"
        eye_open : True if EAR above threshold

        Returns
        -------
        command : "FORWARD" | "LEFT" | "RIGHT" | "STOP" |
                  "EMERGENCY STOP - NO EYE" | "PAUSED"
        """
        now = time.time()

        # ── 1. Eye presence / blink logic ─────────────────────────────
        if not eye_open or gaze_dir == "NO_EYE":
            elapsed = now - self._last_eye_time
            self._register_blink(now)

            if elapsed > self.blink_tolerance:
                self.current_command = "EMERGENCY STOP - NO EYE"
                self._history.clear()
                return self.current_command
            else:
                # Short blink → hold last confirmed command (no jerk)
                return self._last_confirmed
        else:
            self._last_eye_time = now

        # ── 2. Double-blink → pause / resume ──────────────────────────
        if self._check_double_blink(now):
            self._paused = not self._paused
            state_str = "PAUSED" if self._paused else "READY"
            print(f"[Logic] Double-blink detected → {state_str}")

        if self._paused:
            self.current_command = "PAUSED"
            return "PAUSED"

        # ── 3. Debounce ───────────────────────────────────────────────
        self._history.append(gaze_dir)

        if len(self._history) == self.hold_frames:
            unique = set(self._history)
            if len(unique) == 1:
                # Stable gaze — confirm the command
                confirmed = list(unique)[0]
                self._last_confirmed = confirmed
                self.current_command = confirmed
                return confirmed

        # Still in debounce window
        self.current_command = "HOLDING..."
        return "HOLDING..."

    def reset(self):
        """Called when user presses 'r' — full restart."""
        self.state           = "READY"
        self._history.clear()
        self._last_confirmed = "STOP"
        self._paused         = False
        self._last_eye_time  = time.time()
        print("[Logic] Reset complete.")

    # ── internal helpers ──────────────────────────────────────────────────

    def _register_blink(self, t):
        self._blink_times.append(t)

    def _check_double_blink(self, now):
        """Returns True if two blinks occurred within the double_blink_win."""
        recent = [t for t in self._blink_times if now - t < self.double_blink_win]
        if len(recent) >= 2:
            self._blink_times.clear()   # consume the event
            return True
        return False

import time
import json
import os
from collections import deque

DEFAULT_CONFIG = {
    "hold_frames":      10,      
    "blink_tolerance":   0.6,    
    "double_blink_win":  1.0,    
    "paused_on_start":  False    
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
        self.state = "READY"   

        self.hold_frames     = cfg["hold_frames"]
        self._history        = deque(maxlen=self.hold_frames)
        self._last_confirmed = "STOP"

        self.blink_tolerance = cfg["blink_tolerance"]
        self._last_eye_time  = time.time()
        self._was_eye_open   = True # BLINK BUG FIX: Edge detection state

        self.double_blink_win = cfg["double_blink_win"]
        self._blink_times     = deque(maxlen=5)
        self._paused          = cfg["paused_on_start"]

        self.current_command  = "STOP"

    def process(self, gaze_dir: str, eye_open: bool) -> str:
        now = time.time()

        # ── 1. Eye presence / blink logic ─────────────────────────────
        if not eye_open or gaze_dir == "NO_EYE":
            elapsed = now - self._last_eye_time
            
            # BLINK BUG FIX: Only trigger on the exact moment it closes
            if self._was_eye_open:
                self._register_blink(now)
                self._was_eye_open = False

            if elapsed > self.blink_tolerance:
                self.current_command = "EMERGENCY STOP - NO EYE"
                self._history.clear()
                return self.current_command
            else:
                return self._last_confirmed
        else:
            self._last_eye_time = now
            self._was_eye_open = True

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
                confirmed = list(unique)[0]
                self._last_confirmed = confirmed
                self.current_command = confirmed
                return confirmed

        # UX IMPROVEMENT: Stop UI flickering by returning last confirmed instead of "HOLDING..."
        self.current_command = self._last_confirmed
        return self._last_confirmed

    def reset(self):
        self.state           = "READY"
        self._history.clear()
        self._last_confirmed = "STOP"
        self._paused         = False
        self._last_eye_time  = time.time()
        self._was_eye_open   = True
        print("[Logic] Reset complete.")

    def _register_blink(self, t):
        self._blink_times.append(t)

    def _check_double_blink(self, now):
        recent = [t for t in self._blink_times if now - t < self.double_blink_win]
        if len(recent) >= 2:
            self._blink_times.clear()   
            return True
        return False

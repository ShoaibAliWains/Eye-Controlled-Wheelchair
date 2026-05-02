import json
import os
from collections import deque

class LogicController:
    def __init__(self, config_path="config.json"):
        # Load settings or use defaults
        self.hold_frames = 5
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                    self.hold_frames = cfg.get("hold_frames", 5)
            except:
                pass

        self._history = deque(maxlen=self.hold_frames)
        self._last_confirmed = "STOP"
        self.state = "READY"
        print("[Logic] Controller Initialized (Blob Tracking Mode) 🧠")

    def process(self, gaze_dir):
        """
        Takes the raw gaze direction from the tracker and smooths it.
        Requires N consecutive identical frames to change direction.
        """
        # If eye is lost, stop immediately for safety
        if gaze_dir == "NO_EYE":
            self._history.clear()
            self._last_confirmed = "STOP"
            return "STOP"

        # Add current gaze to history
        self._history.append(gaze_dir)

        # If we have enough frames and they are all the same, confirm the command
        if len(self._history) == self.hold_frames and len(set(self._history)) == 1:
            self._last_confirmed = self._history[-1]
            return self._last_confirmed

        # Otherwise, keep doing what we were doing (prevents stuttering)
        return self._last_confirmed

    def reset(self):
        self._history.clear()
        self._last_confirmed = "STOP"

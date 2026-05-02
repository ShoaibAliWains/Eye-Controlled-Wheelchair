import time
from collections import deque

class LogicController:
    def __init__(self, fps=30, hold_time=0.4, blink_tolerance=0.6):
        self.state = "READY" # No more calibrating phase, instantly ready
        self.calibration_samples = [] # Kept for compatibility with main.py
        
        # Hardcoded to the exact center of a 640x480 camera frame
        self.center_x = 320
        self.center_y = 240
        
        # Slightly widened the safe zone for better resting comfort
        self.x_threshold = 30
        self.y_threshold = 25
        
        self.history_length = int(fps * hold_time)
        self.command_history = deque(maxlen=self.history_length)
        
        self.last_eye_time = time.time()
        self.blink_tolerance = blink_tolerance # Grace period before E-STOP

    def get_command(self, pupil_pos):
        # Force state to READY even if main.py tries to reset it
        self.state = "READY"

        # 1. BLINK TOLERANCE & E-STOP
        if pupil_pos is None:
            time_missing = time.time() - self.last_eye_time
            if time_missing > self.blink_tolerance:
                self.command_history.clear()
                return "EMERGENCY STOP - NO EYE"
            else:
                # If short blink, return the last known intended command to prevent jerking
                return "HOLDING..." if len(self.command_history) == 0 else self.command_history[-1]
                
        self.last_eye_time = time.time()

        # 2. DIRECTION MAPPING (Based on fixed center)
        px, py = pupil_pos
        raw_cmd = "STOP"
        if py < self.center_y - self.y_threshold:
            raw_cmd = "FORWARD"
        elif px < self.center_x - self.x_threshold:
            raw_cmd = "LEFT"
        elif px > self.center_x + self.x_threshold:
            raw_cmd = "RIGHT"

        self.command_history.append(raw_cmd)

        # 3. DEBOUNCE
        if len(self.command_history) == self.history_length and len(set(self.command_history)) == 1:
            return raw_cmd
            
        if "STOP" in list(self.command_history)[-4:]:
            return "STOP" # Prioritize user trying to stop

        return "HOLDING..."

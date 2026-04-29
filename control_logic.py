import time
import numpy as np
from collections import deque

class LogicController:
    def __init__(self, fps=30, hold_time=0.4, blink_tolerance=0.6):
        self.state = "INIT"
        self.center_x, self.center_y = 0, 0
        self.calibration_samples = []
        
        # UPDATED: Lowered thresholds to make Left/Right detection much easier
        self.x_threshold = 15
        self.y_threshold = 18
        
        self.history_length = int(fps * hold_time)
        self.command_history = deque(maxlen=self.history_length)
        
        self.last_eye_time = time.time()
        self.blink_tolerance = blink_tolerance # Grace period before E-STOP

    def process_calibration(self, pupil_pos):
        self.state = "CALIBRATING - LOOK CENTER"
        self.calibration_samples.append(pupil_pos)
        
        if len(self.calibration_samples) >= 45: # 1.5 seconds at 30fps
            # Reject blinks/zeros during calibration
            clean_samples = [p for p in self.calibration_samples if p is not None]
            
            if len(clean_samples) > 20:
                x_vals = [p[0] for p in clean_samples]
                y_vals = [p[1] for p in clean_samples]
                # Median filters out wild outliers better than mean
                self.center_x = int(np.median(x_vals))
                self.center_y = int(np.median(y_vals))
                self.state = "READY"
            else:
                self.calibration_samples.clear() # Restart if too noisy
        return "STOP"

    def get_command(self, pupil_pos):
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

        # 2. CALIBRATION ROUTINE
        if self.state != "READY":
            return self.process_calibration(pupil_pos)

        # 3. DIRECTION MAPPING
        px, py = pupil_pos
        raw_cmd = "STOP"
        if py < self.center_y - self.y_threshold:
            raw_cmd = "FORWARD"
        elif px < self.center_x - self.x_threshold:
            raw_cmd = "LEFT"
        elif px > self.center_x + self.x_threshold:
            raw_cmd = "RIGHT"

        self.command_history.append(raw_cmd)

        # 4. DEBOUNCE
        if len(self.command_history) == self.history_length and len(set(self.command_history)) == 1:
            return raw_cmd
            
        if "STOP" in list(self.command_history)[-4:]:
            return "STOP" # Prioritize user trying to stop

        return "HOLDING..."

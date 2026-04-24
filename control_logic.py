from collections import deque

class LogicController:
    def __init__(self, fps=30, hold_time=0.5):
        self.calibrated = False
        self.center_x = 0
        self.center_y = 0
        self.calibration_samples = []
        
        self.x_threshold = 15
        self.y_threshold = 15
        
        self.history_length = int(fps * hold_time)
        self.command_history = deque(maxlen=self.history_length)

    def calibrate(self, pupil_pos):
        if len(self.calibration_samples) < 50:
            self.calibration_samples.append(pupil_pos)
            return "CALIBRATING... LOOK CENTER"
        else:
            x_vals = [p[0] for p in self.calibration_samples]
            y_vals = [p[1] for p in self.calibration_samples]
            self.center_x = sum(x_vals) / len(x_vals)
            self.center_y = sum(y_vals) / len(y_vals)
            self.calibrated = True
            return "CALIBRATION COMPLETE"

    def get_raw_direction(self, pupil_pos):
        if not pupil_pos:
            return "STOP"

        px, py = pupil_pos
        if py < self.center_y - self.y_threshold:
            return "FORWARD"
        elif px < self.center_x - self.x_threshold:
            return "LEFT"
        elif px > self.center_x + self.x_threshold:
            return "RIGHT"
        else:
            return "STOP"

    def get_filtered_command(self, pupil_pos):
        if not self.calibrated:
            return self.calibrate(pupil_pos)

        raw_cmd = self.get_raw_direction(pupil_pos)
        self.command_history.append(raw_cmd)

        if len(self.command_history) == self.history_length and len(set(self.command_history)) == 1:
            return raw_cmd
        elif "STOP" in list(self.command_history)[-3:]: 
            return "STOP"
            
        return "HOLDING..."

"""
motor_control.py  —  H-Bridge Motor Controller
===============================================
Based on working v1. Key fix: LEFT and RIGHT had SAME GPIO states in v1.
Now uses a lookup table so each direction has correct pin states.

Wiring (IN_IN_PWM, forward-only):
  FORWARD : L(in1=H, in2=L)  R(in1=L, in2=H)
  LEFT    : L(in1=L, in2=L)  R(in1=L, in2=H)  ← left motor OFF
  RIGHT   : L(in1=H, in2=L)  R(in1=L, in2=L)  ← right motor OFF
  STOP    : L(in1=L, in2=L)  R(in1=L, in2=L)
"""

import RPi.GPIO as GPIO
import time

# Direction → (L_in1, L_in2, R_in1, R_in2)
_LOGIC = {
    "FORWARD": (True,  False, False, True),
    "LEFT":    (False, False, False, True),
    "RIGHT":   (True,  False, False, False),
    "STOP":    (False, False, False, False),
}


class MotorController:
    def __init__(self, config_l, config_r, driver_type="IN_IN_PWM", max_speed=60):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self.driver_type = driver_type
        self.max_speed   = max_speed
        self.config_l    = config_l
        self.config_r    = config_r

        for cfg in [config_l, config_r]:
            for key, pin in cfg.items():
                if pin is not None:
                    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                    if key == 'en':
                        GPIO.output(pin, GPIO.HIGH)

        self.pwm_l = GPIO.PWM(config_l['pwm'], 1000)
        self.pwm_r = GPIO.PWM(config_r['pwm'], 1000)
        self.pwm_l.start(0)
        self.pwm_r.start(0)

        self.current_speed_l = 0.0
        self.current_speed_r = 0.0
        self.target_speed_l  = 0.0
        self.target_speed_r  = 0.0

        self.accel_step = 1.5
        self.brake_step = 3.0

        self.current_direction = "STOP"
        self.target_direction  = "STOP"

        print(f"[Motors] Ready  driver={driver_type}  max={max_speed}")

    def _apply_logic(self, direction):
        li1, li2, ri1, ri2 = _LOGIC.get(direction, _LOGIC["STOP"])
        if self.driver_type == "IN_IN_PWM":
            GPIO.output(self.config_l['in1'], li1)
            GPIO.output(self.config_l['in2'], li2)
            GPIO.output(self.config_r['in1'], ri1)
            GPIO.output(self.config_r['in2'], ri2)
        elif self.driver_type == "DIR_PWM":
            GPIO.output(self.config_l['dir'], li1)
            GPIO.output(self.config_r['dir'], ri1)

    def set_target(self, direction, speed=None):
        if direction not in _LOGIC:
            direction = "STOP"
        spd = self.max_speed if speed is None else min(speed, self.max_speed)
        self.target_direction = direction

        if direction == "FORWARD":
            self.target_speed_l = spd
            self.target_speed_r = spd
        elif direction == "LEFT":
            self.target_speed_l = 0.0    # pivot: stop left
            self.target_speed_r = spd
        elif direction == "RIGHT":
            self.target_speed_l = spd
            self.target_speed_r = 0.0    # pivot: stop right
        else:
            self.target_speed_l = 0.0
            self.target_speed_r = 0.0

    def update(self):
        """Smooth ramping with safe direction-change (same as v1 logic)."""
        changing = self.current_direction != self.target_direction

        if changing and (self.current_speed_l > 0 or self.current_speed_r > 0):
            # Brake to zero first before flipping GPIO
            self.target_speed_l = 0.0
            self.target_speed_r = 0.0

        elif self.current_speed_l == 0.0 and self.current_speed_r == 0.0:
            if changing:
                self.current_direction = self.target_direction
                self._apply_logic(self.current_direction)
            # Restore real targets after direction flip
            if self.current_direction == "FORWARD":
                self.target_speed_l = self.max_speed
                self.target_speed_r = self.max_speed
            elif self.current_direction == "LEFT":
                self.target_speed_l = 0.0
                self.target_speed_r = self.max_speed
            elif self.current_direction == "RIGHT":
                self.target_speed_l = self.max_speed
                self.target_speed_r = 0.0

        # Ramp
        self.current_speed_l = self._ramp(self.current_speed_l, self.target_speed_l)
        self.current_speed_r = self._ramp(self.current_speed_r, self.target_speed_r)

        self.pwm_l.ChangeDutyCycle(self.current_speed_l)
        self.pwm_r.ChangeDutyCycle(self.current_speed_r)

    def _ramp(self, cur, tgt):
        if cur < tgt:
            return min(cur + self.accel_step, tgt)
        if cur > tgt:
            return max(cur - self.brake_step, tgt)
        return cur

    def emergency_stop(self):
        self.target_speed_l = self.current_speed_l = 0.0
        self.target_speed_r = self.current_speed_r = 0.0
        self.target_direction = self.current_direction = "STOP"
        self.pwm_l.ChangeDutyCycle(0)
        self.pwm_r.ChangeDutyCycle(0)
        self._apply_logic("STOP")

    def cleanup(self):
        self.emergency_stop()
        try:
            self.pwm_l.stop()
            self.pwm_r.stop()
        except Exception:
            pass
        GPIO.cleanup()
        print("[Motors] GPIO cleaned up.")

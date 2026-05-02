import RPi.GPIO as GPIO
import time

# ════════════════ CALIBRATION BLOCK ════════════════
# Wiring issues are handled in main.py pin mapping.
SWAP_LEFT_RIGHT    = False  
INVERT_LEFT_MOTOR  = False  
INVERT_RIGHT_MOTOR = False  
# ═══════════════════════════════════════════════════

class MotorController:
    def __init__(self, config_l, config_r, driver_type="IN_IN_PWM", max_speed=60):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self.driver_type = driver_type
        self.max_speed = max_speed 
        self.config_l = config_l
        self.config_r = config_r

        for cfg in [self.config_l, self.config_r]:
            for key, pin in cfg.items():
                if pin is not None:
                    GPIO.setup(pin, GPIO.OUT)
                    if key == 'en': GPIO.output(pin, GPIO.HIGH)

        self.pwm_l = GPIO.PWM(self.config_l['pwm'], 1000)
        self.pwm_r = GPIO.PWM(self.config_r['pwm'], 1000)
        self.pwm_l.start(0)
        self.pwm_r.start(0)

        self.current_speed_l = 0.0
        self.current_speed_r = 0.0
        self.target_speed_l = 0.0
        self.target_speed_r = 0.0
        
        self.accel_step = 1.5 
        self.brake_step = 3.0 

        self.current_direction = "STOP"
        self.target_direction = "STOP"

    def _apply_logic(self, left_fwd, left_rev, right_fwd, right_rev):
        if SWAP_LEFT_RIGHT:
            left_fwd, left_rev, right_fwd, right_rev = right_fwd, right_rev, left_fwd, left_rev
        if INVERT_LEFT_MOTOR:
            left_fwd, left_rev = left_rev, left_fwd
        if INVERT_RIGHT_MOTOR:
            right_fwd, right_rev = right_rev, right_fwd

        if self.driver_type == "IN_IN_PWM":
            GPIO.output(self.config_l['in1'], left_fwd)
            GPIO.output(self.config_l['in2'], left_rev)
            GPIO.output(self.config_r['in1'], right_fwd)
            GPIO.output(self.config_r['in2'], right_rev)
        elif self.driver_type == "DIR_PWM":
            GPIO.output(self.config_l['dir'], left_fwd)
            GPIO.output(self.config_r['dir'], right_fwd)

    def set_target(self, direction, speed=None):
        if speed is None:
            speed = self.max_speed
        else:
            speed = min(speed, self.max_speed)

        self.target_direction = direction

        if direction == "FORWARD":
            self.target_speed_l = speed
            self.target_speed_r = speed
        elif direction == "LEFT":
            self.target_speed_l = 0     # Stop left wheel
            self.target_speed_r = speed # Drive right wheel
        elif direction == "RIGHT":
            self.target_speed_l = speed # Drive left wheel
            self.target_speed_r = 0     # Stop right wheel
        else: # STOP
            self.target_speed_l = 0
            self.target_speed_r = 0

    def update(self):
        if self.current_direction != self.target_direction and (self.current_speed_l > 0 or self.current_speed_r > 0):
            self.target_speed_l = 0
            self.target_speed_r = 0
        elif self.current_speed_l == 0 and self.current_speed_r == 0:
            self.current_direction = self.target_direction
            
            # Differential Steering: All movements use Forward pins physically, speeds dictate the turn
            if self.current_direction in ["FORWARD", "LEFT", "RIGHT"]:
                self._apply_logic(True, False, True, False)

        for side in ['l', 'r']:
            current = getattr(self, f"current_speed_{side}")
            target = getattr(self, f"target_speed_{side}")
            
            if current < target:
                setattr(self, f"current_speed_{side}", min(current + self.accel_step, target))
            elif current > target:
                setattr(self, f"current_speed_{side}", max(current - self.brake_step, target))

        self.pwm_l.ChangeDutyCycle(self.current_speed_l)
        self.pwm_r.ChangeDutyCycle(self.current_speed_r)

    def emergency_stop(self):
        self.target_speed_l = 0
        self.target_speed_r = 0
        self.current_speed_l = 0
        self.current_speed_r = 0
        self.pwm_l.ChangeDutyCycle(0)
        self.pwm_r.ChangeDutyCycle(0)
        self._apply_logic(False, False, False, False)

    def cleanup(self):
        self.emergency_stop()
        try:
            self.pwm_l.stop()
            self.pwm_r.stop()
        except:
            pass
        GPIO.cleanup()

import RPi.GPIO as GPIO
import time

class MotorController:
    def __init__(self, config_l, config_r, driver_type="IN_IN_PWM", max_speed=60):
        """
        driver_type can be "IN_IN_PWM" or "DIR_PWM"
        """
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self.driver_type = driver_type
        self.max_speed = max_speed # Hard safety limit (0-100)
        self.config_l = config_l
        self.config_r = config_r

        # Setup all pins provided in the configs
        for cfg in [self.config_l, self.config_r]:
            for key, pin in cfg.items():
                if pin is not None:
                    GPIO.setup(pin, GPIO.OUT)
                    if key == 'en': GPIO.output(pin, GPIO.HIGH) # Enable by default

        # Initialize PWM (1kHz)
        self.pwm_l = GPIO.PWM(self.config_l['pwm'], 1000)
        self.pwm_r = GPIO.PWM(self.config_r['pwm'], 1000)
        self.pwm_l.start(0)
        self.pwm_r.start(0)

        # State Tracking
        self.current_speed_l = 0.0
        self.current_speed_r = 0.0
        self.target_speed_l = 0.0
        self.target_speed_r = 0.0
        
        # Acceleration/Deceleration factor (Soft braking)
        self.accel_step = 1.5 
        self.brake_step = 3.0 # Braking is faster than accelerating for safety

        self.current_direction = "STOP"
        self.target_direction = "STOP"

    def _apply_logic(self, left_fwd, left_rev, right_fwd, right_rev):
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

        # DIFFERENTIAL DRIVE FIX: Both motors spin, but turning side spins slower
        if direction == "FORWARD":
            self.target_speed_l = speed
            self.target_speed_r = speed
        elif direction == "LEFT":
            self.target_speed_l = speed * 0.3 # Slow down left motor to turn left smoothly
            self.target_speed_r = speed
        elif direction == "RIGHT":
            self.target_speed_l = speed
            self.target_speed_r = speed * 0.3 # Slow down right motor to turn right smoothly
        else: # STOP
            self.target_speed_l = 0
            self.target_speed_r = 0

    def update(self):
        """Must be called in main loop for smooth ramping and safe direction changes."""
        
        # 1. STOP BEFORE REVERSE SAFETY
        # If changing from FWD to REV, force speed to 0 first before flipping relays/logic
        if self.current_direction != self.target_direction and (self.current_speed_l > 0 or self.current_speed_r > 0):
            self.target_speed_l = 0
            self.target_speed_r = 0
        elif self.current_speed_l == 0 and self.current_speed_r == 0:
            # Once stopped, it is safe to change the physical hardware logic
            self.current_direction = self.target_direction
            
            # EXPLICIT STOP & FIX: Both motors stay in FORWARD state for turns based on client wiring
            if self.current_direction in ["FORWARD", "LEFT", "RIGHT"]:
                self._apply_logic(True, False, False, True)
            elif self.current_direction == "STOP":
                self._apply_logic(False, False, False, False)

        # 2. SOFT ACCELERATION & BRAKING
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
        """Instant halt for critical failures."""
        self.target_speed_l = 0
        self.target_speed_r = 0
        self.current_speed_l = 0
        self.current_speed_r = 0
        self.pwm_l.ChangeDutyCycle(0)
        self.pwm_r.ChangeDutyCycle(0)
        self.current_direction = "STOP"
        self._apply_logic(False, False, False, False)

    def cleanup(self):
        self.emergency_stop()
        GPIO.cleanup()

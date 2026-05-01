"""
motor_control.py  —  H-Bridge Motor Controller  (v3)
=====================================================

Key fixes over v2
-----------------
* LEFT and RIGHT now have their own correct GPIO pin states
  (v2 applied identical logic for all three directions — LEFT & RIGHT
   would have turned the same way!)
* emergency_stop() now also sets target_direction = "STOP" so update()
  does not re-apply motion on the next tick
* cleanup() calls GPIO.cleanup() only after stop — safe for KeyboardInterrupt
* Configurable via constructor kwargs; no hidden globals

Wiring assumed  (IN_IN_PWM mode, single forward direction)
----------------------------------------------------------
  FORWARD : L→(in1=HIGH, in2=LOW)   R→(in1=LOW,  in2=HIGH)
  LEFT    : L→(in1=LOW,  in2=LOW )  R→(in1=LOW,  in2=HIGH)   ← L motor off
  RIGHT   : L→(in1=HIGH, in2=LOW )  R→(in1=LOW,  in2=LOW )   ← R motor off
  STOP    : L→(in1=LOW,  in2=LOW )  R→(in1=LOW,  in2=LOW )
"""

import RPi.GPIO as GPIO
import time


# GPIO states as named constants for readability
_H = GPIO.HIGH
_L = GPIO.LOW

# Direction → (L_in1, L_in2, R_in1, R_in2)
_LOGIC_TABLE = {
    "FORWARD": (_H, _L, _L, _H),
    "LEFT":    (_L, _L, _L, _H),   # left motor coast, right motor drives
    "RIGHT":   (_H, _L, _L, _L),   # left motor drives, right motor coast
    "STOP":    (_L, _L, _L, _L),
}


class MotorController:
    def __init__(self,
                 config_l,
                 config_r,
                 driver_type="IN_IN_PWM",
                 max_speed=75,
                 accel_step=1.5,
                 brake_step=3.0,
                 pwm_freq=1000):
        """
        Parameters
        ----------
        config_l / config_r : dict with keys
            in1, in2  — direction pins  (IN_IN_PWM mode)
            pwm       — PWM-capable pin
            dir       — direction pin   (DIR_PWM mode only)
            en        — enable pin      (optional, pulled HIGH on init)
            (set unused keys to None)
        driver_type : "IN_IN_PWM" or "DIR_PWM"
        max_speed   : hard duty-cycle ceiling  (0–100)
        accel_step  : duty-cycle increase per update() tick
        brake_step  : duty-cycle decrease per update() tick  (faster = safer)
        pwm_freq    : PWM frequency in Hz
        """
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self.driver_type = driver_type
        self.max_speed   = max_speed
        self.config_l    = config_l
        self.config_r    = config_r
        self.accel_step  = accel_step
        self.brake_step  = brake_step

        # Setup all configured pins
        for cfg in (config_l, config_r):
            for key, pin in cfg.items():
                if pin is not None:
                    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                    if key == "en":
                        GPIO.output(pin, GPIO.HIGH)   # enable driver chip

        # Start PWM at 0% duty
        self.pwm_l = GPIO.PWM(config_l["pwm"], pwm_freq)
        self.pwm_r = GPIO.PWM(config_r["pwm"], pwm_freq)
        self.pwm_l.start(0)
        self.pwm_r.start(0)

        # Speed state
        self.current_speed_l = 0.0
        self.current_speed_r = 0.0
        self.target_speed_l  = 0.0
        self.target_speed_r  = 0.0

        # Direction state
        self.current_direction = "STOP"
        self.target_direction  = "STOP"

        print(f"[Motors] Initialized  |  driver={driver_type}  max_speed={max_speed}")

    # ── public API ────────────────────────────────────────────────────────

    def set_target(self, direction: str, speed: float = None):
        """
        Queue a new target direction.  Speed ramps on each update() call.
        direction : "FORWARD" | "LEFT" | "RIGHT" | "STOP"
        speed     : duty-cycle override (capped at max_speed)
        """
        if direction not in _LOGIC_TABLE:
            direction = "STOP"

        spd = self.max_speed if speed is None else min(speed, self.max_speed)
        self.target_direction = direction

        if direction == "FORWARD":
            self.target_speed_l = spd
            self.target_speed_r = spd
        elif direction == "LEFT":
            # Zero-radius pivot: stop left wheel, drive right
            self.target_speed_l = 0.0
            self.target_speed_r = spd
        elif direction == "RIGHT":
            # Zero-radius pivot: drive left, stop right
            self.target_speed_l = spd
            self.target_speed_r = 0.0
        else:  # STOP
            self.target_speed_l = 0.0
            self.target_speed_r = 0.0

    def update(self):
        """
        Call every frame in the main loop.
        Handles:
          1. Safe direction change: ramp to zero before flipping GPIO pins
          2. Soft acceleration / braking ramp
          3. PWM duty-cycle update
        """
        direction_changing = (self.current_direction != self.target_direction)

        # ── Step 1: If direction changes, brake to zero first ─────────
        if direction_changing and (self.current_speed_l > 0 or
                                   self.current_speed_r > 0):
            # Override targets to zero until motors stop
            self.target_speed_l = 0.0
            self.target_speed_r = 0.0

        elif self.current_speed_l == 0.0 and self.current_speed_r == 0.0:
            # Motors are stopped — safe to flip GPIO direction pins
            if direction_changing:
                self.current_direction = self.target_direction
                self._apply_logic(self.current_direction)

            # Now restore the real speed targets for the new direction
            if self.current_direction == "FORWARD":
                self.target_speed_l = self.max_speed
                self.target_speed_r = self.max_speed
            elif self.current_direction == "LEFT":
                self.target_speed_l = 0.0
                self.target_speed_r = self.max_speed
            elif self.current_direction == "RIGHT":
                self.target_speed_l = self.max_speed
                self.target_speed_r = 0.0
            # STOP → targets remain 0.0

        # ── Step 2: Ramp each motor toward its target ─────────────────
        self.current_speed_l = self._ramp(
            self.current_speed_l, self.target_speed_l)
        self.current_speed_r = self._ramp(
            self.current_speed_r, self.target_speed_r)

        # ── Step 3: Write duty cycles ──────────────────────────────────
        self.pwm_l.ChangeDutyCycle(self.current_speed_l)
        self.pwm_r.ChangeDutyCycle(self.current_speed_r)

    def emergency_stop(self):
        """Instant halt — bypasses all ramping.  Call on E-STOP / KeyboardInterrupt."""
        self.target_speed_l    = 0.0
        self.target_speed_r    = 0.0
        self.current_speed_l   = 0.0
        self.current_speed_r   = 0.0
        self.target_direction  = "STOP"
        self.current_direction = "STOP"
        self.pwm_l.ChangeDutyCycle(0)
        self.pwm_r.ChangeDutyCycle(0)
        self._apply_logic("STOP")

    def cleanup(self):
        """Call in finally block.  Stops motors then releases GPIO."""
        self.emergency_stop()
        try:
            self.pwm_l.stop()
            self.pwm_r.stop()
        except Exception:
            pass
        GPIO.cleanup()
        print("[Motors] GPIO cleaned up.")

    # ── internals ─────────────────────────────────────────────────────────

    def _ramp(self, current: float, target: float) -> float:
        if current < target:
            return min(current + self.accel_step, target)
        elif current > target:
            return max(current - self.brake_step, target)
        return current

    def _apply_logic(self, direction: str):
        """Write GPIO pin states for the given direction."""
        li1, li2, ri1, ri2 = _LOGIC_TABLE.get(direction, _LOGIC_TABLE["STOP"])

        if self.driver_type == "IN_IN_PWM":
            GPIO.output(self.config_l["in1"], li1)
            GPIO.output(self.config_l["in2"], li2)
            GPIO.output(self.config_r["in1"], ri1)
            GPIO.output(self.config_r["in2"], ri2)
        elif self.driver_type == "DIR_PWM":
            # DIR_PWM: HIGH = forward, LOW = reverse
            GPIO.output(self.config_l["dir"], li1)
            GPIO.output(self.config_r["dir"], ri1)

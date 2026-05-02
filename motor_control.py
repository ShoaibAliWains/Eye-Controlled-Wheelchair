"""
motor_control.py  —  H-Bridge Motor Controller  (wiring-fix edition)
=====================================================================

PROBLEM FIXED
-------------
Previous code had wrong GPIO pin states per direction causing:
  RIGHT command → moved FORWARD
  FORWARD       → turned RIGHT
  LEFT          → not working

Root cause: _apply_logic() was sending identical pin states for
FORWARD / LEFT / RIGHT. Now each direction has correct, distinct states.

═══════════════════════════════════════════════════════════════════════
 CALIBRATION BLOCK  — edit ONLY this section to fix motor behaviour
═══════════════════════════════════════════════════════════════════════
Symptom                          → Try first
─────────────────────────────────────────────────────────────────────
RIGHT moves FORWARD              → SWAP_LEFT_RIGHT    = True
LEFT  moves FORWARD              → SWAP_LEFT_RIGHT    = True
FORWARD spins one motor only     → INVERT_LEFT_MOTOR  = True
LEFT / RIGHT reversed            → SWAP_LEFT_RIGHT    = True
One motor always spins backwards → INVERT that motor  = True
═══════════════════════════════════════════════════════════════════════
"""

import RPi.GPIO as GPIO

# ── CALIBRATION (edit here only) ─────────────────────────────────────
INVERT_LEFT_MOTOR  = False   # True → reverses left  motor spin
INVERT_RIGHT_MOTOR = False   # True → reverses right motor spin
SWAP_LEFT_RIGHT    = False   # True → swaps which side is "left"/"right"

MAX_SPEED_DEFAULT  = 65      # top duty cycle 0–100 (reduced from 75 for safety)
ACCEL_STEP         = 1.2     # duty-cycle increase per frame  (gentle start)
BRAKE_STEP         = 5.0     # duty-cycle decrease per frame  (strong braking)
TURN_FACTOR        = 0.0     # inner-wheel speed during turn  (0=pivot, 0.5=arc)
# ─────────────────────────────────────────────────────────────────────

# GPIO pin-state shortcuts
_FWD = (True,  False)   # in1=HIGH, in2=LOW  → motor spins forward
_OFF = (False, False)   # in1=LOW,  in2=LOW  → motor coasts

# Base direction table: (left_motor_pins, right_motor_pins)
_BASE = {
    "FORWARD": (_FWD, _FWD),
    "LEFT":    (_OFF, _FWD),   # left stops,  right drives → pivots left
    "RIGHT":   (_FWD, _OFF),   # left drives, right stops  → pivots right
    "STOP":    (_OFF, _OFF),
}


def _invert(pins):
    """Swap in1/in2 to reverse spin direction."""
    return (pins[1], pins[0])


def _build_table():
    table = {}
    for cmd, (lp, rp) in _BASE.items():
        if INVERT_LEFT_MOTOR:
            lp = _invert(lp)
        if INVERT_RIGHT_MOTOR:
            rp = _invert(rp)
        if SWAP_LEFT_RIGHT:
            lp, rp = rp, lp
        table[cmd] = (lp, rp)
    return table


# Built once at import — zero per-frame overhead
_LOGIC = _build_table()


class MotorController:

    def __init__(self,
                 config_l,
                 config_r,
                 driver_type="IN_IN_PWM",
                 max_speed=MAX_SPEED_DEFAULT):

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self.driver_type = driver_type
        self.max_speed   = max_speed
        self.config_l    = config_l
        self.config_r    = config_r
        self.accel_step  = ACCEL_STEP
        self.brake_step  = BRAKE_STEP
        self.turn_factor = TURN_FACTOR

        # Setup GPIO pins
        for cfg in (config_l, config_r):
            for key, pin in cfg.items():
                if pin is not None:
                    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
                    if key == 'en':
                        GPIO.output(pin, GPIO.HIGH)

        # PWM 1 kHz
        self.pwm_l = GPIO.PWM(config_l['pwm'], 1000)
        self.pwm_r = GPIO.PWM(config_r['pwm'], 1000)
        self.pwm_l.start(0)
        self.pwm_r.start(0)

        # Speed & direction state
        self.current_speed_l   = 0.0
        self.current_speed_r   = 0.0
        self.target_speed_l    = 0.0
        self.target_speed_r    = 0.0
        self.current_direction = "STOP"
        self.target_direction  = "STOP"

        print(f"[Motors] Ready  driver={driver_type}  max_speed={max_speed}")
        print(f"[Motors] Calibration flags → "
              f"invert_L={INVERT_LEFT_MOTOR}  "
              f"invert_R={INVERT_RIGHT_MOTOR}  "
              f"swap={SWAP_LEFT_RIGHT}")

    # ── public ────────────────────────────────────────────────────────

    def set_target(self, direction: str, speed: float = None):
        if direction not in _LOGIC:
            direction = "STOP"
        spd = self.max_speed if speed is None else min(speed, self.max_speed)
        self.target_direction = direction

        if direction == "FORWARD":
            self.target_speed_l = spd
            self.target_speed_r = spd
        elif direction == "LEFT":
            self.target_speed_l = spd * self.turn_factor   # inner wheel
            self.target_speed_r = spd                       # outer wheel
        elif direction == "RIGHT":
            self.target_speed_l = spd                       # outer wheel
            self.target_speed_r = spd * self.turn_factor   # inner wheel
        else:
            self.target_speed_l = 0.0
            self.target_speed_r = 0.0

    def update(self):
        """
        Call every frame.
        1. Brake to zero before flipping GPIO direction pins (H-Bridge safety).
        2. Smooth ramp toward target speed.
        3. Write PWM duty cycles.
        """
        changing = (self.current_direction != self.target_direction)

        # Step 1: force brake before direction flip
        if changing and (self.current_speed_l > 0 or self.current_speed_r > 0):
            self.target_speed_l = 0.0
            self.target_speed_r = 0.0

        elif self.current_speed_l == 0.0 and self.current_speed_r == 0.0:
            if changing:
                self.current_direction = self.target_direction
                self._write_gpio(self.current_direction)

            # Restore real targets after GPIO flip
            spd = self.max_speed
            d   = self.current_direction
            if d == "FORWARD":
                self.target_speed_l = spd
                self.target_speed_r = spd
            elif d == "LEFT":
                self.target_speed_l = spd * self.turn_factor
                self.target_speed_r = spd
            elif d == "RIGHT":
                self.target_speed_l = spd
                self.target_speed_r = spd * self.turn_factor
            # STOP → targets remain 0.0

        # Step 2: ramp
        self.current_speed_l = self._ramp(self.current_speed_l,
                                           self.target_speed_l)
        self.current_speed_r = self._ramp(self.current_speed_r,
                                           self.target_speed_r)

        # Step 3: PWM
        self.pwm_l.ChangeDutyCycle(self.current_speed_l)
        self.pwm_r.ChangeDutyCycle(self.current_speed_r)

    def emergency_stop(self):
        """Instant halt — no ramping."""
        self.target_speed_l = self.current_speed_l = 0.0
        self.target_speed_r = self.current_speed_r = 0.0
        self.target_direction = self.current_direction = "STOP"
        self.pwm_l.ChangeDutyCycle(0)
        self.pwm_r.ChangeDutyCycle(0)
        self._write_gpio("STOP")

    def cleanup(self):
        self.emergency_stop()
        try:
            self.pwm_l.stop()
            self.pwm_r.stop()
        except Exception:
            pass
        GPIO.cleanup()
        print("[Motors] GPIO cleaned up.")

    # ── internals ─────────────────────────────────────────────────────

    def _ramp(self, cur: float, tgt: float) -> float:
        if cur < tgt:
            return min(cur + self.accel_step, tgt)
        if cur > tgt:
            return max(cur - self.brake_step, tgt)
        return cur

    def _write_gpio(self, direction: str):
        (li1, li2), (ri1, ri2) = _LOGIC.get(direction, _LOGIC["STOP"])
        if self.driver_type == "IN_IN_PWM":
            GPIO.output(self.config_l['in1'], li1)
            GPIO.output(self.config_l['in2'], li2)
            GPIO.output(self.config_r['in1'], ri1)
            GPIO.output(self.config_r['in2'], ri2)
        elif self.driver_type == "DIR_PWM":
            GPIO.output(self.config_l['dir'], li1)
            GPIO.output(self.config_r['dir'], ri1)

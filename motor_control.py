import RPi.GPIO as GPIO

class MotorController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        # Pin definitions for dual BTS7960
        self.L_FWD_PIN = 12
        self.L_REV_PIN = 13
        self.R_FWD_PIN = 18
        self.R_REV_PIN = 19

        # Setup pins
        pins = [self.L_FWD_PIN, self.L_REV_PIN, self.R_FWD_PIN, self.R_REV_PIN]
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)

        # Initialize PWM (1kHz frequency)
        self.pwm_l_fwd = GPIO.PWM(self.L_FWD_PIN, 1000)
        self.pwm_l_rev = GPIO.PWM(self.L_REV_PIN, 1000)
        self.pwm_r_fwd = GPIO.PWM(self.R_FWD_PIN, 1000)
        self.pwm_r_rev = GPIO.PWM(self.R_REV_PIN, 1000)

        self._start_pwm()
        self.stop()

    def _start_pwm(self):
        self.pwm_l_fwd.start(0)
        self.pwm_l_rev.start(0)
        self.pwm_r_fwd.start(0)
        self.pwm_r_rev.start(0)

    def move_forward(self, speed=50):
        self.pwm_l_fwd.ChangeDutyCycle(speed)
        self.pwm_l_rev.ChangeDutyCycle(0)
        self.pwm_r_fwd.ChangeDutyCycle(speed)
        self.pwm_r_rev.ChangeDutyCycle(0)

    def turn_left(self, speed=40):
        self.pwm_l_fwd.ChangeDutyCycle(0)
        self.pwm_l_rev.ChangeDutyCycle(speed)
        self.pwm_r_fwd.ChangeDutyCycle(speed)
        self.pwm_r_rev.ChangeDutyCycle(0)

    def turn_right(self, speed=40):
        self.pwm_l_fwd.ChangeDutyCycle(speed)
        self.pwm_l_rev.ChangeDutyCycle(0)
        self.pwm_r_fwd.ChangeDutyCycle(0)
        self.pwm_r_rev.ChangeDutyCycle(speed)

    def stop(self):
        self.pwm_l_fwd.ChangeDutyCycle(0)
        self.pwm_l_rev.ChangeDutyCycle(0)
        self.pwm_r_fwd.ChangeDutyCycle(0)
        self.pwm_r_rev.ChangeDutyCycle(0)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()

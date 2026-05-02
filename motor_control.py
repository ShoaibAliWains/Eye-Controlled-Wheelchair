import RPi.GPIO as GPIO

class MotorController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        self.left_pins = {'in1': 5, 'in2': 6, 'pwm': 12}
        self.right_pins = {'in1': 16, 'in2': 20, 'pwm': 13}
        self.max_speed = 75
        
        for pins in [self.left_pins, self.right_pins]:
            GPIO.setup(pins['in1'], GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(pins['in2'], GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(pins['pwm'], GPIO.OUT)
            
        self.pwm_l = GPIO.PWM(self.left_pins['pwm'], 1000)
        self.pwm_r = GPIO.PWM(self.right_pins['pwm'], 1000)
        self.pwm_l.start(0)
        self.pwm_r.start(0)

    def set_target(self, direction):
        if direction == "FORWARD":
            self._set_pins(self.left_pins, 1, 0, self.max_speed)
            self._set_pins(self.right_pins, 0, 1, self.max_speed)
        elif direction == "LEFT":
            self._set_pins(self.left_pins, 0, 0, 0)
            self._set_pins(self.right_pins, 0, 1, self.max_speed)
        elif direction == "RIGHT":
            self._set_pins(self.left_pins, 1, 0, self.max_speed)
            self._set_pins(self.right_pins, 0, 0, 0)
        else: # STOP
            self._set_pins(self.left_pins, 0, 0, 0)
            self._set_pins(self.right_pins, 0, 0, 0)

    def _set_pins(self, pins, in1, in2, speed):
        GPIO.output(pins['in1'], in1)
        GPIO.output(pins['in2'], in2)
        pins_pwm = self.pwm_l if pins == self.left_pins else self.pwm_r
        pins_pwm.ChangeDutyCycle(speed)

    def cleanup(self):
        self.set_target("STOP")
        self.pwm_l.stop()
        self.pwm_r.stop()
        GPIO.cleanup()

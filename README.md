# Eye-Controlled Wheelchair System (Production Release)

An embedded computer-vision system utilizing Raspberry Pi 4, `libcamera`, and a configurable dual H-Bridge.

## ⚠️ Safety Features Included
* **Soft Braking:** Motor deceleration prevents throwing the user forward.
* **Reversal Safety:** The system explicitly requires motors to hit 0 RPM before changing direction logic, preventing H-Bridge burnout.
* **Blink Resilience:** A 0.6-second grace period ignores natural blinking.
* **Emergency Halt:** Automatically stops if the eye is lost for >0.6 seconds.
* **CPU Optimization:** Loop sleeping prevents the Pi 4 from overheating.

## Hardware Setup
You can use any 12V-24V Dual Motor Driver. Open `main.py` and modify the Configuration section:
* Set `DRIVER_TYPE` to `"IN_IN_PWM"` for L298N/Generic logic.
* Set `DRIVER_TYPE` to `"DIR_PWM"` for Cytron/BTS7960 logic.
* Update the `LEFT_MOTOR` and `RIGHT_MOTOR` pin dictionaries with your Pi's GPIO numbers. Put `None` for pins you are not using.

## Running the Code
Ensure you are using Raspberry Pi OS Bullseye/Bookworm with Picamera2 installed.
```bash
python3 main.py

Calibration & Operation
Boot Phase: Put the glasses on. Ensure a clear view of the eye.

Calibration: Look straight ahead at your natural resting angle. The screen will say CALIBRATING - LOOK CENTER. Hold still for 1.5 seconds.

Ready State: A blue bounding box will appear on the HUD. This is your "Neutral Deadzone".

Movement: Look outside the blue box to trigger movement. Look back inside the box to brake.

Recalibrate: Press r on the keyboard at any time to reset the deadzone.


---

### **💡 Explanation of Improvements & Why They Matter**

1.  **Configurable Motor Drivers:** Instead of hardcoding BTS7960 logic, I built a dictionary-based configuration (`LEFT_MOTOR` / `RIGHT_MOTOR`) and a `driver_type` toggle. This means if your client uses a Cytron MDD10A or an L298N, they don't have to rewrite the GPIO logic; they just change one word in `main.py`.
2.  **Stop-Before-Reverse Safety:** *This is critical for real hardware.* If a heavy wheelchair is moving forward and the user looks left, abruptly flipping the reverse pin on one motor while it still has forward momentum will pull massive current spikes and fry the H-Bridge. I updated `motor_control.py` to enforce a hard rule: `If target_direction != current_direction -> Drop speed to 0 first -> Then flip pins -> Then accelerate.`
3.  **Temporal Smoothing (EMA):** Haar cascades natively jitter back and forth by a few pixels every frame. I added an Exponential Moving Average (`self.smooth_x`, `self.alpha`) in `eye_tracking.py`. This smooths the pupil tracking, drastically reducing micro-stutters in the user control logic.
4.  **Blink Tolerance vs. E-Stop:** Previously, a 0.1-second blink would instantly trigger a stop command, making the ride incredibly jerky. Now, the `LogicController` measures `time_missing`. If the eye vanishes for under 0.6 seconds, it "holds" the last valid command. If it exceeds 0.6s, it triggers an absolute Emergency Stop. 
5.  **CPU Yielding:** Running an infinite `while True` loop on a Pi 4 with OpenCV will max out one CPU core to 100%, causing thermal throttling. I added a frame delta check (`elapsed`) that forces the loop to `time.sleep()` for a few milliseconds, capping the system at a stable 30-33 FPS, keeping the Pi cool.

import cv2
import time
from camera import Camera
from eye_tracking import EyeTracker
from control_logic import LogicController
from motor_control import MotorController

# ================= CONFIGURATION =================
# Set your driver type: "IN_IN_PWM" or "DIR_PWM"
DRIVER_TYPE = "IN_IN_PWM" 

# Pin Definitions (Set unused to None)
LEFT_MOTOR = {'in1': 5, 'in2': 6, 'pwm': 12, 'dir': None, 'en': None}
RIGHT_MOTOR = {'in1': 16, 'in2': 20, 'pwm': 13, 'dir': None, 'en': None}

MAX_SPEED_LIMIT = 55 # 0-100 safe speed
# =================================================

def draw_ui(frame, logic, command, pupil_pos, motors):
    # Draw Safe Zone if calibrated
    if logic.state == "READY":
        pt1 = (logic.center_x - logic.x_threshold, logic.center_y - logic.y_threshold)
        pt2 = (logic.center_x + logic.x_threshold, logic.center_y + logic.y_threshold)
        cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)
        cv2.circle(frame, (logic.center_x, logic.center_y), 2, (255, 0, 0), -1)

    # Draw Pupil
    if pupil_pos:
        cv2.line(frame, (pupil_pos[0]-10, pupil_pos[1]), (pupil_pos[0]+10, pupil_pos[1]), (0, 255, 0), 2)
        cv2.line(frame, (pupil_pos[0], pupil_pos[1]-10), (pupil_pos[0], pupil_pos[1]+10), (0, 255, 0), 2)

    # Info HUD
    cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1) # Black background for text
    cv2.putText(frame, f"STATE: {logic.state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    color = (0, 0, 255) if "STOP" in command else (0, 255, 0)
    cv2.putText(frame, f"CMD: {command}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.putText(frame, f"PWR L:{motors.current_speed_l:.0f} R:{motors.current_speed_r:.0f}", 
                (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    print("Initializing Core Systems...")
    cam = Camera()
    tracker = EyeTracker()
    logic = LogicController()
    motors = MotorController(LEFT_MOTOR, RIGHT_MOTOR, DRIVER_TYPE, MAX_SPEED_LIMIT)

    try:
        while True:
            loop_start = time.time()

            frame = cam.get_frame()
            if frame is None:
                continue

            pupil_pos, display_frame = tracker.get_pupil_position(frame)
            command = logic.get_command(pupil_pos)
            
            # Target Setting
            if "STOP" in command or logic.state != "READY":
                motors.set_target("STOP")
            elif command in ["FORWARD", "LEFT", "RIGHT"]:
                motors.set_target(command)

            # Ramping & Execution
            motors.update()

            # UI Update
            draw_ui(display_frame, logic, command, pupil_pos, motors)
            cv2.imshow("Wheelchair Vision HUD", display_frame)

            # Keyboard Input & FPS Cap
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                logic.state = "INIT"
                logic.calibration_samples.clear()
                motors.emergency_stop()

            # Performance Optimization: CPU Yield
            # Calculate time taken, sleep to maintain ~30 FPS
            elapsed = time.time() - loop_start
            if elapsed < 0.03:
                time.sleep(0.03 - elapsed)

    except KeyboardInterrupt:
        print("\nManual Override Triggered.")
    finally:
        print("Safely powering down H-Bridges...")
        motors.cleanup()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import time
from camera import Camera
from eye_tracking import EyeTracker
from motor_control import MotorController
from control_logic import LogicController

def main():
    cam = Camera()
    tracker = EyeTracker()
    motors = MotorController()
    logic = LogicController()

    print("System Running: Original Eye-Box AI Restored")

    try:
        while True:
            frame = cam.get_frame()
            if frame is None: continue

            # Get tracking data (Original format)
            gaze_dir, eye_open, display = tracker.get_gaze(frame)
            
            # Smooth the command
            command = logic.process(gaze_dir)
            
            # Send to motors
            motors.set_target(command)

            # Command text on HUD
            cmd_color = (0, 255, 0) if command in ["FORWARD", "LEFT", "RIGHT"] else (0, 0, 255)
            cv2.putText(display, f"CMD: {command}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, cmd_color, 3)

            cv2.imshow("Original Tracker HUD", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        motors.cleanup()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

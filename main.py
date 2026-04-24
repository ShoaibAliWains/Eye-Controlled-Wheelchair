import cv2
from camera import Camera
from eye_tracking import EyeTracker
from control_logic import LogicController
from motor_control import MotorController

def main():
    print("Initializing System...")
    cam = Camera()
    tracker = EyeTracker()
    logic = LogicController(fps=30, hold_time=0.5)
    motors = MotorController()

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            pupil_pos, display_frame = tracker.get_pupil_position(frame)
            
            if pupil_pos:
                cv2.circle(display_frame, pupil_pos, 5, (0, 255, 0), -1)

            if not pupil_pos:
                motors.stop()
                command = "NO EYE DETECTED - STOP"
                logic.command_history.clear() 
            else:
                command = logic.get_filtered_command(pupil_pos)
                
                if command == "FORWARD":
                    motors.move_forward(speed=50)
                elif command == "LEFT":
                    motors.turn_left(speed=40)
                elif command == "RIGHT":
                    motors.turn_right(speed=40)
                elif command == "STOP":
                    motors.stop()

            # Display telemetry
            cv2.putText(display_frame, f"CMD: {command}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if logic.calibrated:
                cv2.circle(display_frame, (int(logic.center_x), int(logic.center_y)), 
                           logic.x_threshold, (255, 0, 0), 1)

            cv2.imshow("Eye Wheelchair Interface", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                logic.calibrated = False
                logic.calibration_samples = []

    except KeyboardInterrupt:
        print("\nSystem interrupted by user.")
    finally:
        print("Cleaning up GPIO and Camera...")
        motors.cleanup()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
from camera import Camera
from eye_tracking import EyeTracker
from motor_control import MotorController
from control_logic import LogicController

def main():
    cam = Camera()
    tracker = EyeTracker()
    motors = MotorController()
    logic = LogicController()

    try:
        while True:
            frame = cam.get_frame()
            if frame is None: continue

            gaze_dir, eye_open, display = tracker.get_gaze(frame)
            command = logic.process(gaze_dir, eye_open)
            
            if command in ["FORWARD", "LEFT", "RIGHT", "STOP"]:
                motors.set_target(command)
            
            motors.update()
            cv2.imshow("Final Eye Tracker", display)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        motors.cleanup()
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

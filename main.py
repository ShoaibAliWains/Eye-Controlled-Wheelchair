import cv2
import time
import numpy as np
from camera import Camera
from eye_tracking import EyeTracker
from motor_control import MotorController

def main():
    cam = Camera()
    tracker = EyeTracker()
    motors = MotorController()

    print("System Running: Direct Blob Tracking (Video Mode)")

    try:
        while True:
            frame = cam.get_frame()
            if frame is None: continue

            # Get tracking data
            gaze_dir, thresh_view, display = tracker.get_gaze(frame)
            
            # Send to motors
            motors.set_target(gaze_dir)

            # --- HUD like the Video ---
            # Shrink threshold view to show in corner
            th, tw = thresh_view.shape[:2]
            thresh_small = cv2.resize(thresh_view, (tw//3, th//3))
            sth, stw = thresh_small.shape[:2]
            
            # Overlay threshold view on top left
            display[0:sth, 0:stw] = thresh_small
            cv2.rectangle(display, (0, 0), (stw, sth), (0, 255, 0), 2)
            cv2.putText(display, "AI SCAN", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Command text
            cmd_color = (0, 255, 0) if gaze_dir in ["FORWARD", "LEFT", "RIGHT"] else (0, 0, 255)
            cv2.putText(display, f"CMD: {gaze_dir}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, cmd_color, 3)

            cv2.imshow("Direct Tracking HUD", display)

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

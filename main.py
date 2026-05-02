"""
main.py  —  Eye-Controlled Wheelchair
======================================
Structure kept identical to working v1.
Only changes:
  • tracker.get_gaze()    instead of tracker.get_pupil_position()
  • logic.get_command(gaze_dir, eye_open)  instead of get_command(pupil_pos)
  • Iron-Man HUD corners + semi-transparent status bar
  • r key now calls logic.reset() properly
"""

import cv2
import time
from camera        import Camera
from eye_tracking  import EyeTracker
from control_logic import LogicController
from motor_control import MotorController

# ═══════════════════════════ CONFIGURATION ═══════════════════════════════════
DRIVER_TYPE = "IN_IN_PWM"          # "IN_IN_PWM" or "DIR_PWM"

LEFT_MOTOR  = {'in1': 5,  'in2': 6,  'pwm': 12, 'dir': None, 'en': None}
RIGHT_MOTOR = {'in1': 16, 'in2': 20, 'pwm': 13, 'dir': None, 'en': None}

MAX_SPEED = 75     # 0-100
TARGET_FPS = 30
# ═════════════════════════════════════════════════════════════════════════════

FRAME_MS = int(1000 / TARGET_FPS)

# Colours (BGR)
_G  = (0,  255,  80)   # green
_R  = (0,   60, 220)   # red
_Y  = (0,  220, 255)   # yellow
_CY = (255, 200,  0)   # cyan
_OR = (0,  165, 255)   # orange
_GR = (120, 120, 120)  # gray
_W  = (220, 220, 220)  # white


def _rect(frame, x, y, w, h, color, alpha=0.6):
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def draw_ui(frame, logic, command, motors, gaze_dir, eye_open):
    fh, fw = frame.shape[:2]

    # Semi-transparent top bar
    _rect(frame, 0, 0, fw, 72, (0, 0, 0), alpha=0.65)

    # STATE
    cv2.putText(frame, f"STATE: {logic.state}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _Y, 2)

    # COMMAND colour
    if "EMERGENCY" in command or "STOP" in command:
        cc = _R
    elif command in ("FORWARD", "LEFT", "RIGHT"):
        cc = _G
    elif command == "PAUSED":
        cc = _OR
    else:
        cc = _GR
    cv2.putText(frame, f"CMD: {command}",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, cc, 2)

    # Motor power (top-right)
    cv2.putText(frame,
                f"PWR L:{motors.current_speed_l:.0f} R:{motors.current_speed_r:.0f}",
                (fw - 220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GR, 1)

    # Eye status
    cv2.putText(frame, "EYE:OPEN" if eye_open else "EYE:CLOSED",
                (fw - 170, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                _G if eye_open else _R, 1)

    # Gaze direction arrow (bottom-centre)
    cx, cy = fw // 2, fh - 38
    arrows = {
        "FORWARD": ((cx, cy + 18), (cx, cy - 18)),
        "LEFT":    ((cx + 18, cy), (cx - 18, cy)),
        "RIGHT":   ((cx - 18, cy), (cx + 18, cy)),
    }
    if gaze_dir in arrows:
        cv2.arrowedLine(frame, *arrows[gaze_dir], _CY, 2, tipLength=0.4)

    # Iron-Man corner brackets
    BL, BT = 20, 2
    for (bx, by), (dh, dv) in zip(
        [(0, 0), (fw - BL, 0), (0, fh - BL), (fw - BL, fh - BL)],
        [(1, 1), (-1, 1),      (1, -1),      (-1, -1)],
    ):
        cv2.line(frame, (bx, by), (bx + dh * BL, by), _G, BT)
        cv2.line(frame, (bx, by), (bx, by + dv * BL), _G, BT)

    # PAUSED overlay
    if command == "PAUSED":
        _rect(frame, 0, 0, fw, fh, (0, 0, 0), alpha=0.35)
        cv2.putText(frame, "-- PAUSED --",
                    (fw // 2 - 115, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, _OR, 3)
        cv2.putText(frame, "double-blink to resume",
                    (fw // 2 - 135, fh // 2 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _W, 1)

    cv2.putText(frame, "q=quit  r=reset",
                (10, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GR, 1)


def main():
    print("=" * 55)
    print("  Eye-Controlled Wheelchair  |  Relative Gaze Mode")
    print("=" * 55)
    print("Initializing Core Systems...")

    cam     = Camera()
    tracker = EyeTracker(config_path="config.json")
    logic   = LogicController(config_path="config.json")
    motors  = MotorController(LEFT_MOTOR, RIGHT_MOTOR, DRIVER_TYPE, MAX_SPEED)

    print("\n[Ready]  Look at the camera.  q=quit  r=reset\n")

    gaze_dir = "NO_EYE"
    eye_open = False
    command  = "STOP"

    try:
        while True:
            loop_start = time.time()

            frame = cam.get_frame()
            if frame is None:
                continue

            # Eye tracking → gaze direction
            gaze_dir, eye_open, display_frame = tracker.get_gaze(frame)

            # Logic → motor command
            command = logic.get_command(gaze_dir, eye_open)

            # Motor control
            if "EMERGENCY" in command or command in ("STOP", "PAUSED"):
                if "EMERGENCY" in command:
                    motors.emergency_stop()
                else:
                    motors.set_target("STOP")
            elif command in ("FORWARD", "LEFT", "RIGHT"):
                motors.set_target(command)
            # "HOLDING..." → keep ramping toward current target

            motors.update()

            # Draw HUD
            draw_ui(display_frame, logic, command, motors, gaze_dir, eye_open)
            cv2.imshow("Wheelchair Vision HUD", display_frame)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[Quit]")
                break
            elif key == ord('r'):
                print("\n[Reset]")
                logic.reset()
                motors.emergency_stop()

            # FPS cap (same as v1)
            elapsed = time.time() - loop_start
            sleep_t = (1.0 / TARGET_FPS) - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Ctrl+C]")
    finally:
        print("Shutting down safely...")
        motors.cleanup()
        cam.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()

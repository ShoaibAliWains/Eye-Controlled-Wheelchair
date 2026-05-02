"""
Eye-Controlled Wheelchair — main.py  (v4  •  Auto-Calibration + Dynamic Pupil)
===============================================================================
Run:   python3 main.py
Stop:  q
Reset: r  (restarts calibration + logic)
"""

import cv2
import time
import numpy as np

from camera        import Camera
from eye_tracking  import EyeTracker
from control_logic import LogicController
from motor_control import MotorController


# ═══════════════════════════════════════════════════════════════════════
#  HARDWARE  — edit only this block
# ═══════════════════════════════════════════════════════════════════════
DRIVER_TYPE = "IN_IN_PWM"
LEFT_MOTOR  = {'in1': 5,  'in2': 6,  'pwm': 12, 'dir': None, 'en': None}
RIGHT_MOTOR = {'in1': 16, 'in2': 20, 'pwm': 13, 'dir': None, 'en': None}
MAX_SPEED   = 75
TARGET_FPS  = 30
# ═══════════════════════════════════════════════════════════════════════

FRAME_BUDGET = 1.0 / TARGET_FPS

_GREEN  = (0,  255,  80)
_RED    = (0,   60, 220)
_YELLOW = (0,  220, 255)
_CYAN   = (255, 220,  0)
_WHITE  = (230, 230, 230)
_GRAY   = (130, 130, 130)
_ORANGE = (0,  165, 255)


def _rect(frame, x, y, w, h, color, alpha=0.6):
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def draw_ui(frame, logic, command, motors, gaze_dir, eye_open):
    fh, fw = frame.shape[:2]
    _rect(frame, 0, 0, fw, 72, (0, 0, 0), alpha=0.65)

    state_col = _YELLOW if logic.state == "READY" else _ORANGE
    cv2.putText(frame, f"STATE: {logic.state}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_col, 2)

    if "EMERGENCY" in command or "STOP" in command:
        cmd_col = _RED
    elif command in ("FORWARD", "LEFT", "RIGHT"):
        cmd_col = _GREEN
    elif command in ("PAUSED", "CALIBRATING..."):
        cmd_col = _ORANGE
    else:
        cmd_col = _GRAY

    cv2.putText(frame, f"CMD: {command}",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, cmd_col, 2)

    pwr = f"PWR  L:{motors.current_speed_l:.0f}  R:{motors.current_speed_r:.0f}"
    cv2.putText(frame, pwr, (fw - 230, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1)
    eye_txt = "EYE: OPEN" if eye_open else "EYE: CLOSED"
    cv2.putText(frame, eye_txt, (fw - 180, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, _GREEN if eye_open else _RED, 1)

    cx, cy = fw // 2, fh - 40
    arrows = {
        "FORWARD": ((cx, cy + 20), (cx, cy - 20)),
        "LEFT":    ((cx + 20, cy), (cx - 20, cy)),
        "RIGHT":   ((cx - 20, cy), (cx + 20, cy)),
    }
    if gaze_dir in arrows:
        cv2.arrowedLine(frame, *arrows[gaze_dir], _CYAN, 2, tipLength=0.4)

    BL, BT = 22, 2
    for (bx, by), (dhx, dhy), (dvx, dvy) in zip(
        [(0, 0), (fw - BL, 0), (0, fh - BL), (fw - BL, fh - BL)],
        [(1, 0), (-1, 0), (1, 0), (-1, 0)],
        [(0, 1), (0, 1), (0, -1), (0, -1)],
    ):
        cv2.line(frame, (bx, by), (bx + dhx * BL, by + dhy * BL), _GREEN, BT)
        cv2.line(frame, (bx, by), (bx + dvx * BL, by + dvy * BL), _GREEN, BT)

    if command == "PAUSED":
        _rect(frame, 0, 0, fw, fh, (0, 0, 0), alpha=0.35)
        cv2.putText(frame, "-- PAUSED --",
                    (fw // 2 - 120, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, _ORANGE, 3)
        cv2.putText(frame, "double-blink to resume",
                    (fw // 2 - 140, fh // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1)

    cv2.putText(frame, "q=quit  r=reset/recalibrate",
                (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GRAY, 1)
    return frame


def main():
    print("=" * 60)
    print("  Eye-Controlled Wheelchair  |  v4  |  Auto-Calibration")
    print("=" * 60)

    cam     = Camera(config_path="config.json")
    tracker = EyeTracker(config_path="config.json")
    logic   = LogicController(config_path="config.json")
    motors  = MotorController(LEFT_MOTOR, RIGHT_MOTOR, DRIVER_TYPE, MAX_SPEED)

    gaze_dir = "NO_EYE"
    eye_open = False
    command  = "STOP"

    print("\n[Main] Look straight at the camera — calibration starting!\n")

    try:
        while True:
            t0 = time.time()

            frame = cam.get_frame()
            if frame is None:
                continue

            gaze_dir, eye_open, display = tracker.get_gaze(frame)
            command = logic.process(gaze_dir, eye_open,
                                    calibrating=tracker.calibrating)

            if "EMERGENCY" in command or command in ("STOP", "CALIBRATING...", "PAUSED"):
                if "EMERGENCY" in command:
                    motors.emergency_stop()
                else:
                    motors.set_target("STOP")
            elif command in ("FORWARD", "LEFT", "RIGHT"):
                motors.set_target(command)

            motors.update()

            display = draw_ui(display, logic, command, motors, gaze_dir, eye_open)
            cv2.imshow("Wheelchair Vision HUD  v4", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("\n[Reset] Restarting calibration …")
                tracker = EyeTracker(config_path="config.json")
                logic.reset()
                motors.emergency_stop()

            elapsed = time.time() - t0
            if elapsed < FRAME_BUDGET:
                time.sleep(FRAME_BUDGET - elapsed)

    except KeyboardInterrupt:
        print("\n[Ctrl+C]")
    finally:
        motors.cleanup()
        cam.release()
        cv2.destroyAllWindows()
        print("Safe.")


if __name__ == "__main__":
    main()

"""
Eye-Controlled Wheelchair — main.py  (v2  •  dlib gaze-ratio edition)
======================================================================
Run:   python3 main.py
Stop:  press  q
Reset: press  r
"""

import cv2
import time
import numpy as np
import json

from camera        import Camera
from eye_tracking  import EyeTracker
from control_logic import LogicController
from motor_control import MotorController


# ═══════════════════════════════════════════════════════════════════════
#  HARDWARE CONFIGURATION  — edit only this section for your wiring
# ═══════════════════════════════════════════════════════════════════════
DRIVER_TYPE  = "IN_IN_PWM"          # "IN_IN_PWM"  or  "DIR_PWM"
LEFT_MOTOR   = {'in1': 5,  'in2': 6,  'pwm': 12, 'dir': None, 'en': None}
RIGHT_MOTOR  = {'in1': 16, 'in2': 20, 'pwm': 13, 'dir': None, 'en': None}

# Load Max Speed dynamically from config.json
try:
    with open('config.json', 'r') as f:
        MAX_SPEED = json.load(f).get("max_motor_speed", 75)
except Exception as e:
    print(f"[WARN] Error loading speed from config.json: {e}")
    MAX_SPEED = 75

TARGET_FPS   = 30
FRAME_BUDGET = 1.0 / TARGET_FPS
# ═══════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────
#  HUD renderer
# ─────────────────────────────────────────────────────────────────────

# Colour palette (BGR)
_C_GREEN   = (0,  255,  80)
_C_RED     = (0,   60, 220)
_C_YELLOW  = (0,  220, 255)
_C_CYAN    = (255, 220,  0)
_C_WHITE   = (230, 230, 230)
_C_GRAY    = (130, 130, 130)
_C_ORANGE  = (0,  165, 255)


def _overlay_rect(frame, x, y, w, h, color, alpha=0.55):
    """Semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_ui(frame, logic, command, motors, gaze_dir, eye_open):
    h, w = frame.shape[:2]

    # ── Top status bar ────────────────────────────────────────────────
    _overlay_rect(frame, 0, 0, w, 70, (0, 0, 0), alpha=0.65)

    # State
    state_color = _C_YELLOW if logic.state == "READY" else _C_ORANGE
    cv2.putText(frame, f"STATE: {logic.state}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

    # Command
    if "STOP" in command or "EMERGENCY" in command:
        cmd_color = _C_RED
    elif command in ("FORWARD", "LEFT", "RIGHT"):
        cmd_color = _C_GREEN
    elif command == "PAUSED":
        cmd_color = _C_ORANGE
    else:
        cmd_color = _C_GRAY

    cv2.putText(frame, f"CMD: {command}",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, cmd_color, 2)

    # Motor power (top-right)
    pwr_txt = f"PWR  L:{motors.current_speed_l:.0f}  R:{motors.current_speed_r:.0f}"
    cv2.putText(frame, pwr_txt,
                (w - 230, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_GRAY, 1)

    # Eye status indicator (top-right, second line)
    eye_txt   = "EYE: OPEN" if eye_open else "EYE: CLOSED"
    eye_color = _C_GREEN   if eye_open else _C_RED
    cv2.putText(frame, eye_txt,
                (w - 180, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)

    # ── Gaze direction arrow  (bottom centre) ────────────────────────
    cx = w // 2
    cy = h - 40

    arrow_map = {
        "FORWARD": ((cx, cy + 20), (cx, cy - 20)),
        "LEFT":    ((cx + 20, cy), (cx - 20, cy)),
        "RIGHT":   ((cx - 20, cy), (cx + 20, cy)),
    }
    if gaze_dir in arrow_map:
        p1, p2 = arrow_map[gaze_dir]
        cv2.arrowedLine(frame, p1, p2, _C_CYAN, 2, tipLength=0.4)

    # ── Corner brackets  (Iron Man frame style) ──────────────────────
    bracket_color = _C_GREEN
    blen, bthk = 22, 2
    corners = [(0, 0), (w - blen, 0), (0, h - blen), (w - blen, h - blen)]
    dirs_h  = [(1, 0), (-1, 0), (1, 0), (-1, 0)]   # horizontal direction
    dirs_v  = [(0, 1), (0, 1), (0, -1), (0, -1)]   # vertical direction

    for (cx2, cy2), (dhx, dhy), (dvx, dvy) in zip(corners, dirs_h, dirs_v):
        cv2.line(frame,
                 (cx2, cy2),
                 (cx2 + dhx * blen, cy2 + dhy * blen),
                 bracket_color, bthk)
        cv2.line(frame,
                 (cx2, cy2),
                 (cx2 + dvx * blen, cy2 + dvy * blen),
                 bracket_color, bthk)

    # ── PAUSED overlay ────────────────────────────────────────────────
    if command == "PAUSED":
        _overlay_rect(frame, 0, 0, w, h, (0, 0, 0), alpha=0.35)
        cv2.putText(frame, "-- PAUSED --",
                    (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, _C_ORANGE, 3)
        cv2.putText(frame, "double-blink to resume",
                    (w // 2 - 140, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _C_WHITE, 1)

    # ── Key hints  (bottom-left) ──────────────────────────────────────
    cv2.putText(frame, "q=quit  r=reset",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _C_GRAY, 1)

    return frame


# ─────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Eye-Controlled Wheelchair  |  v2  |  dlib edition")
    print("=" * 55)
    print("Initializing subsystems …\n")

    cam     = Camera()
    tracker = EyeTracker(predictor_path="shape_predictor_68_face_landmarks.dat")
    logic   = LogicController(config_path="config.json")
    motors  = MotorController(LEFT_MOTOR, RIGHT_MOTOR, DRIVER_TYPE, MAX_SPEED)

    print("\n[Ready]  Look at the camera.  Press  r  to reset, q  to quit.\n")

    gaze_dir = "NO_EYE"
    eye_open = False
    command  = "STOP"

    try:
        while True:
            t0 = time.time()

            # 1. Grab frame
            frame = cam.get_frame()
            if frame is None:
                continue

            # 2. Eye tracking  (returns gaze string + annotated frame)
            gaze_dir, eye_open, display_frame = tracker.get_gaze(frame)

            # 3. Logic  → command
            command = logic.process(gaze_dir, eye_open)

            # 4. Motor control
            if "EMERGENCY" in command or "STOP" in command or command == "PAUSED":
                if "EMERGENCY" in command:
                    motors.emergency_stop()
                else:
                    motors.set_target("STOP")
            elif command in ("FORWARD", "LEFT", "RIGHT"):
                motors.set_target(command)
            # "HOLDING..." → keep current target (smooth ramp)

            motors.update()

            # 5. Draw HUD
            display_frame = draw_ui(
                display_frame, logic, command, motors, gaze_dir, eye_open
            )

            cv2.imshow("Wheelchair Vision HUD", display_frame)

            # 6. Key input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[Quit] User requested exit.")
                break
            elif key == ord('r'):
                print("\n[Reset] Restarting logic …")
                logic.reset()
                motors.emergency_stop()
                
                # Reload speed settings live from JSON!
                try:
                    with open('config.json', 'r') as f:
                        new_speed = json.load(f).get("max_motor_speed", 75)
                        motors.max_speed = new_speed
                except:
                    pass

            # 7. FPS cap
            elapsed = time.time() - t0
            if elapsed < FRAME_BUDGET:
                time.sleep(FRAME_BUDGET - elapsed)

    except KeyboardInterrupt:
        print("\n[Manual Override]  Ctrl+C received.")
    finally:
        print("\nShutting down …")
        motors.cleanup()
        cam.release()
        cv2.destroyAllWindows()
        print("Systems offline. Safe.")


if __name__ == "__main__":
    main()

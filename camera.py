"""
camera.py  —  Raspberry Pi Camera Module  (v3  •  glasses-optimised)
=====================================================================

Changes from v1
---------------
* ROI crop support: if the camera is zoomed in too much, software-crop
  to a centred region so the eye fills ~70% of the frame (ideal for Haar)
* Contrast / brightness boost option for dim environments (glasses shadow)
* Configurable resolution and flip axis via config.json
* Frame warmup: discards first N frames so auto-exposure settles
"""

import cv2
import json
import os
import time
from picamera2 import Picamera2


_DEFAULTS = {
    "resolution":       [640, 480],   # [width, height]
    "framerate":        30,
    "flip_horizontal":  True,         # mirror for intuitive HUD
    "flip_vertical":    False,
    # Software crop: set to null to disable, or [x, y, w, h] in pixels
    "crop_roi":         None,
    # Brightness / contrast tweak (applied with cv2.convertScaleAbs)
    "alpha":            1.0,          # contrast  (1.0 = no change)
    "beta":             0,            # brightness (0   = no change)
    "warmup_frames":    20,           # frames to discard on startup
}


def _load_config(path="config.json"):
    cfg = _DEFAULTS.copy()
    if os.path.exists(path):
        try:
            with open(path) as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"[Camera] Config error ({e}) — using defaults")
    return cfg


class Camera:
    def __init__(self, config_path="config.json"):
        cfg = _load_config(config_path)

        self._res        = tuple(cfg["resolution"])          # (w, h)
        self._flip_h     = cfg["flip_horizontal"]
        self._flip_v     = cfg["flip_vertical"]
        self._crop       = cfg["crop_roi"]                   # None or [x,y,w,h]
        self._alpha      = float(cfg["alpha"])
        self._beta       = int(cfg["beta"])
        self._warmup     = int(cfg["warmup_frames"])

        self.picam2 = Picamera2()
        video_cfg = self.picam2.create_video_configuration(
            main={"size": self._res, "format": "RGB888"},
            controls={"FrameRate": cfg["framerate"]},
        )
        self.picam2.configure(video_cfg)
        self.picam2.start()

        # Let auto-exposure settle
        print(f"[Camera] Warming up ({self._warmup} frames) …")
        for _ in range(self._warmup):
            self.picam2.capture_array()
        print(f"[Camera] Ready  |  res={self._res}  flip_h={self._flip_h}")

    # ── public ────────────────────────────────────────────────────────────

    def get_frame(self):
        """
        Returns a processed BGR frame ready for eye tracking, or None on error.
        Pipeline: capture → BGR convert → flip → crop → enhance
        """
        try:
            frame = self.picam2.capture_array()            # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # → BGR

            # Flip
            if self._flip_h and self._flip_v:
                frame = cv2.flip(frame, -1)
            elif self._flip_h:
                frame = cv2.flip(frame, 1)
            elif self._flip_v:
                frame = cv2.flip(frame, 0)

            # Optional software crop  (helps when camera is too zoomed in)
            if self._crop:
                x, y, w, h = self._crop
                frame = frame[y:y + h, x:x + w]

            # Optional brightness / contrast boost
            if self._alpha != 1.0 or self._beta != 0:
                frame = cv2.convertScaleAbs(frame,
                                            alpha=self._alpha,
                                            beta=self._beta)

            return frame

        except Exception as e:
            print(f"[Camera][WARN] Frame dropped: {e}")
            return None

    def release(self):
        try:
            self.picam2.stop()
            self.picam2.close()
            print("[Camera] Released.")
        except Exception:
            pass

import cv2
from picamera2 import Picamera2

class Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        video_cfg = self.picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        self.picam2.configure(video_cfg)
        self.picam2.start()
        # Warmup: let auto-exposure settle
        for _ in range(10):
            self.picam2.capture_array()
        print("[Camera] Ready  |  640x480 @ 30fps")

    def get_frame(self):
        try:
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return cv2.flip(frame, 1)  # Mirror for HUD
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

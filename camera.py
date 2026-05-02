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
        # Warmup
        for _ in range(10): self.picam2.capture_array()

    def get_frame(self):
        try:
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return cv2.flip(frame, 1) # Mirror for HUD
        except:
            return None

    def release(self):
        self.picam2.stop()
        self.picam2.close()

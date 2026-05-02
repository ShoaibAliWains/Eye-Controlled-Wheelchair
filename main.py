from picamera2 import Picamera2
import cv2

class Camera:
    def __init__(self, resolution=(640, 480), framerate=30):
        self.picam2 = Picamera2()
        
        # Configure for low-latency RGB stream
        config = self.picam2.create_video_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        print("Hardware Camera Initialized (libcamera).")

    def get_frame(self):
        try:
            # Direct memory access to sensor array (zero-copy if optimized)
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return cv2.flip(frame, 1) # Intuitive mirroring
        except Exception as e:
            print(f"[WARN] Camera frame dropped: {e}")
            return None

    def release(self):
        self.picam2.stop()
        self.picam2.close()

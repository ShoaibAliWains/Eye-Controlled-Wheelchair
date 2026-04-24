import cv2

class Camera:
    def __init__(self, camera_index=0, resolution=(640, 480)):
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not initialize camera.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1) # Mirror image for intuitive control

    def release(self):
        self.cap.release()

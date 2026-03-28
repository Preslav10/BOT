import cv2
import threading
import time

class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.running = True

        threading.Thread(target=self.update, daemon=True).start()

        # 🔥 ВАЖНО: изчаква първия frame
        while self.frame is None:
            time.sleep(0.01)

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()
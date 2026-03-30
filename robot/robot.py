import sys
import os

# 👉 FIX за module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import logging
import time

from vision.detector import detect
from vision.depth import estimate_depth
from memory.object_memory import ObjectMemory
from ai.behavior import BehaviorSystem
from voice.text_to_speech import speak
from vision.face_recognition_module import FaceRecognizer

logging.basicConfig(level=logging.INFO)


class Robot:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.memory = ObjectMemory()
        self.behavior = BehaviorSystem()

        # 👤 Face Recognition
        self.face_recognizer = FaceRecognizer()

        # 🧠 Learning control
        self.last_saved_time = 0
        self.SAVE_INTERVAL = 5  # секунди

    def perceive(self):
        ret, frame = self.cap.read()

        if not ret:
            logging.warning("Failed to read frame from camera")
            return None, None

        detections = detect(frame)
        depth_map = estimate_depth(frame)
        faces = self.face_recognizer.recognize(frame)

        return frame, (detections, depth_map, faces)

    def save_new_face(self, face_img):
        if face_img is None:
            return

        current_time = time.time()

        if current_time - self.last_saved_time < self.SAVE_INTERVAL:
            return

        filename = f"faces/person_{int(time.time())}.jpg"

        try:
            if face_img.dtype != 'uint8':
                face_img = (face_img * 255).astype("uint8")

            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(filename, face_img)
            logging.info(f"New face saved: {filename}")

            self.last_saved_time = current_time

        except Exception as e:
            logging.error(f"Error saving face: {e}")

    def think(self, detections, depth_map, faces):
        enriched_objects = []

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]

            h, w = depth_map.shape[:2]
            x1 = max(0, min(int(x1), w - 1))
            x2 = max(0, min(int(x2), w))
            y1 = max(0, min(int(y1), h - 1))
            y2 = max(0, min(int(y2), h))

            roi = depth_map[y1:y2, x1:x2]

            if roi.size == 0:
                depth = 0
            else:
                depth = float(roi.mean())

            obj = {
                "label": label,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "distance": depth   # 🔥 важно: behavior очаква distance
            }

            self.memory.update_object(
                label=obj["label"],
                position=obj["bbox"],
                depth=obj["distance"]
            )

            enriched_objects.append(obj)

        # 👤 learning (без говорене!)
        for face in faces:
            if face["name"] == "Unknown":
                self.save_new_face(face["face_img"])

        # ✅ ВАЖНО: подаваме faces към behavior
        decision = self.behavior.decide(enriched_objects, faces, self.memory)

        return decision, enriched_objects

    def act(self, decision):
        if decision is None:
            return

        logging.info(f"Decision: {decision}")

        action = decision.get("action")

        if action == "speak":
            speak(decision.get("text", ""))

        elif action == "move":
            logging.info(f"Moving: {decision.get('state')}")

    def draw_objects(self, frame, objects):
        for obj in objects:
            x, y, w, h = obj["bbox"]
            label = obj["label"]
            depth = obj["distance"]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            text = f"{label} {depth:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(frame, (x, y - th - 10), (x + tw, y), (255, 0, 0), -1)

            cv2.putText(
                frame,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    def draw_faces(self, frame, faces):
        h_frame, w_frame, _ = frame.shape

        for face in faces:
            x1, y1, x2, y2 = face["box"]

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w_frame, int(x2))
            y2 = min(h_frame, int(y2))

            name = face["name"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)

            cv2.putText(
                frame,
                name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

    def run(self):
        while True:
            frame, data = self.perceive()

            if frame is None:
                continue

            detections, depth_map, faces = data

            decision, objects = self.think(detections, depth_map, faces)
            self.act(decision)

            self.draw_objects(frame, objects)
            self.draw_faces(frame, faces)

            cv2.putText(
                frame,
                f"STATE: {self.behavior.state}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow("Robot Vision", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.shutdown()

    def shutdown(self):
        logging.info("Shutting down robot...")
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    robot = Robot()
    robot.run()
import cv2
import logging

from vision.detector import detect
from vision.depth import estimate_depth
from memory.object_memory import ObjectMemory
from ai.behavior import BehaviorSystem
from voice.text_to_speech import speak

logging.basicConfig(level=logging.INFO)


class Robot:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.memory = ObjectMemory()
        self.behavior = BehaviorSystem()

    def perceive(self):
        ret, frame = self.cap.read()

        if not ret:
            logging.warning("Failed to read frame from camera")
            return None, None

        detections = detect(frame)
        depth_map = estimate_depth(frame)

        return frame, (detections, depth_map)

    def think(self, detections, depth_map):
        enriched_objects = []

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]

            # 🛡️ безопасно изрязване
            h, w = depth_map.shape[:2]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            roi = depth_map[y1:y2, x1:x2]

            # 🛡️ защита от празни региони
            if roi.size == 0:
                depth = 0
            else:
                depth = float(roi.mean())

            obj = {
                "label": label,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "depth": depth
            }

            # 🧠 запис в паметта
            self.memory.update_object(
                label=obj["label"],
                position=obj["bbox"],
                depth=obj["depth"]
            )

            enriched_objects.append(obj)

        # 🧠 behavior system
        decision = self.behavior.update(enriched_objects, self.memory)

        return decision, enriched_objects

    def act(self, decision):
        if decision is None:
            return

        logging.info(f"Decision: {decision}")

        action = decision.get("action")

        if action == "speak":
            speak(decision.get("text", ""))

        elif action == "move":
            # TODO: реално движение (Arduino / motors)
            logging.info("Moving robot...")

    def run(self):
        while True:
            frame, data = self.perceive()

            if frame is None:
                continue

            detections, depth_map = data

            decision, objects = self.think(detections, depth_map)
            self.act(decision)

            # 🎨 визуализация
            for obj in objects:
                x, y, w, h = obj["bbox"]
                label = obj["label"]
                depth = obj["depth"]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {depth:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            # 🧠 показване на state
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
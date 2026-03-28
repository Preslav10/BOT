import cv2

from utils.camera import CameraStream
from vision.detector import detect
from vision.tracker import extract_feature, match_object, object_memory

from vision.face_recognition_module import recognize, save_new_face

from ai.decision import decide, learn
from memory.brain import brain

from ai.training import train_object_model
from voice.text_to_speech import speak
from voice.speech_to_text import listen


CLASSES = ["person", "object"]


# =========================
# DISTANCE (basic)
# =========================
def estimate_distance(w, h):
    return 1000 / (w + h + 1e-6)


# =========================
# MAIN LOOP
# =========================
def main():
    cam = CameraStream(0)

    print("[INFO] AI Robot started...")

    while True:
        frame = cam.read()
        if frame is None:
            continue

        detections = detect(frame)

        for det in detections:
            conf = det[4]
            if conf < 0.5:
                continue

            x, y, w, h = det[:4]

            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # =========================
            # OBJECT FEATURE
            # =========================
            feat = extract_feature(crop)
            obj_id = match_object(feat)
            object_memory[obj_id].append(feat)

            label = CLASSES[0]  # TODO: replace with real YOLO class
            dist = estimate_distance(w, h)

            # =========================
            # FACE RECOGNITION
            # =========================
            faces = recognize(frame)
            person_name = None

            for name, (top, right, bottom, left) in faces:
                person_name = name

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # =========================
            # BRAIN (IDENTIFY)
            # =========================
            known_person = person_name if person_name else "unknown"
            known_obj = brain.identify(None, feat)[1]

            # =========================
            # DECISION (DQN)
            # =========================
            action, state, action_idx = decide(label, dist, conf)

            # =========================
            # ACTIVE LEARNING
            # =========================
            if known_person == "unknown" or known_obj == "unknown":

                speak("Какво е това?")
                name = listen()

                if name:
                    if label == "person":
                        save_new_face(name, crop)
                        speak(f"Запомних те като {name}")
                    else:
                        brain.learn_object(name, feat)
                        speak(f"Запомних обекта като {name}")
                        train_object_model()

            else:
                speak(f"Виждам {known_person or known_obj}")

            # =========================
            # REWARD (for DQN)
            # =========================
            reward = 0

            if label == "person" and dist < 100:
                reward = 1 if action == "STOP" else -0.5
            else:
                reward = 0.2

            next_state = state
            learn(state, action_idx, reward, next_state)

            # =========================
            # DRAW OBJECT
            # =========================
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame,
                        f"{label} ID:{obj_id} {action}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

        cv2.imshow("AI ROBOT", frame)

        if cv2.waitKey(1) == 27:
            break

    cam.stop()
    cv2.destroyAllWindows()


# =========================
if __name__ == "__main__":
    main()
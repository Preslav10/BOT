import os
import cv2
import numpy as np

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except:
    DEEPFACE_AVAILABLE = False


FACES_DIR = "faces"


class FaceRecognition:
    def __init__(self):
        self.known_faces = []
        self.known_names = []

        self.load_faces()

    def load_faces(self):
        if not os.path.exists(FACES_DIR):
            os.makedirs(FACES_DIR)

        for filename in os.listdir(FACES_DIR):
            path = os.path.join(FACES_DIR, filename)

            img = cv2.imread(path)
            if img is None:
                continue

            name = os.path.splitext(filename)[0]

            self.known_faces.append(img)
            self.known_names.append(name)

    def recognize(self, frame):
        results = []

        if not DEEPFACE_AVAILABLE:
            return results

        try:
            detections = DeepFace.extract_faces(frame, enforce_detection=False)
        except:
            return results

        for det in detections:
            face_img = det["face"]
            face_img = (face_img * 255).astype(np.uint8)

            name = self.match_face(face_img)

            results.append({
                "name": name,
                "face_img": face_img
            })

        return results

    def match_face(self, face_img):
        best_match = "unknown"

        for known_img, name in zip(self.known_faces, self.known_names):
            try:
                result = DeepFace.verify(face_img, known_img, enforce_detection=False)

                if result["verified"]:
                    return name
            except:
                continue

        return best_match

    def save_face(self, face_img, name):
        path = os.path.join(FACES_DIR, f"{name}.jpg")
        cv2.imwrite(path, face_img)

        self.known_faces.append(face_img)
        self.known_names.append(name)


# 🔥 Функция за multiprocessing
_face_system = None


def recognize_faces(frame):
    global _face_system

    if _face_system is None:
        _face_system = FaceRecognition()

    return _face_system.recognize(frame)
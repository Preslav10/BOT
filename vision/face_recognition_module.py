import os
import cv2
import numpy as np
from deepface import DeepFace

# Папка за лица
DB_PATH = "faces_db"
os.makedirs(DB_PATH, exist_ok=True)

# =========================
# EXTRACT FACE EMBEDDING
# =========================
def get_face_embedding(frame):
    try:
        result = DeepFace.represent(
            img_path=frame,
            model_name="Facenet",
            enforce_detection=False
        )

        if isinstance(result, list):
            return np.array(result[0]["embedding"])

        return None
    except:
        return None


# =========================
# RECOGNIZE FACE
# =========================
def recognize(frame):
    results = []

    try:
        detections = DeepFace.extract_faces(
            img_path=frame,
            enforce_detection=False
        )

        for det in detections:
            face_img = det["face"]
            x, y, w, h = det["facial_area"].values()

            emb = get_face_embedding(face_img)

            name = "unknown"

            if emb is not None:
                name = match_face(emb)

            results.append((name, (y, x+w, y+h, x)))

    except:
        pass

    return results


# =========================
# SIMPLE MATCH (distance)
# =========================
def match_face(embedding, threshold=0.6):
    best_name = "unknown"
    best_dist = 999

    for person in os.listdir(DB_PATH):
        person_path = os.path.join(DB_PATH, person)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            try:
                ref = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet",
                    enforce_detection=False
                )

                ref_emb = np.array(ref[0]["embedding"])
                dist = np.linalg.norm(embedding - ref_emb)

                if dist < best_dist:
                    best_dist = dist
                    best_name = person

            except:
                continue

    if best_dist < threshold:
        return best_name

    return "unknown"


# =========================
# SAVE NEW PERSON
# =========================
def save_new_face(name, frame):
    person_path = os.path.join(DB_PATH, name)
    os.makedirs(person_path, exist_ok=True)

    count = len(os.listdir(person_path))
    file_path = os.path.join(person_path, f"{count}.jpg")

    cv2.imwrite(file_path, frame)

    print(f"[INFO] Saved new face: {name}")
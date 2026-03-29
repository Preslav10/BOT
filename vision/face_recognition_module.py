import os
import numpy as np
from deepface import DeepFace

DB_PATH = "faces_db"

def load_database():

    db = {}

    for person in os.listdir(DB_PATH):

        path = os.path.join(DB_PATH, person, "embeddings.npy")

        if os.path.exists(path):

            db[person] = np.load(path)

    return db


def cosine(a,b):

    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


def recognize(face_img, db):

    emb = DeepFace.represent(
        face_img,
        model_name="Facenet",
        enforce_detection=False
    )[0]["embedding"]

    emb = np.array(emb)

    best_name=None
    best_score=0

    for name,vecs in db.items():

        for v in vecs:

            score = cosine(emb,v)

            if score>best_score:

                best_score=score
                best_name=name

    if best_score>0.6:
        return best_name,best_score

    return None,best_score
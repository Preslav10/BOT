import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

FACE_DB = "data/faces.pkl"
OBJ_DB = "data/objects.pkl"
MODEL_PATH = "data/model.pkl"

os.makedirs("data", exist_ok=True)

# ===== FACE =====
class FaceDB:
    def __init__(self):
        self.db = self.load()

    def load(self):
        if os.path.exists(FACE_DB):
            return pickle.load(open(FACE_DB, "rb"))
        return {}

    def save(self):
        pickle.dump(self.db, open(FACE_DB, "wb"))

    def add(self, name, emb):
        self.db.setdefault(name, []).append(emb)
        self.save()

    def recognize(self, emb):
        best, dist = "unknown", 999
        for k, v in self.db.items():
            for e in v:
                d = np.linalg.norm(e - emb)
                if d < dist:
                    best, dist = k, d
        return best if dist < 0.6 else "unknown"


# ===== OBJECT =====
class ObjectDB:
    def __init__(self):
        self.X, self.y = [], []
        self.model = None
        self.load()

    def load(self):
        if os.path.exists(OBJ_DB):
            self.X, self.y = pickle.load(open(OBJ_DB, "rb"))
        if os.path.exists(MODEL_PATH):
            self.model = pickle.load(open(MODEL_PATH, "rb"))

    def save(self):
        pickle.dump((self.X, self.y), open(OBJ_DB, "wb"))
        if self.model:
            pickle.dump(self.model, open(MODEL_PATH, "wb"))

    def add(self, label, feat):
        self.X.append(feat)
        self.y.append(label)
        self.save()

    def train(self):
        if len(self.X) < 5:
            return
        self.model = KNeighborsClassifier(3)
        self.model.fit(self.X, self.y)
        self.save()

    def predict(self, feat):
        if not self.model:
            return "unknown"
        return self.model.predict([feat])[0]


face_db = FaceDB()
obj_db = ObjectDB()

def train_new_person(name, emb):
    face_db.add(name, emb)

def recognize_person(emb):
    return face_db.recognize(emb)

def train_new_object(name, feat):
    obj_db.add(name, feat)

def train_object_model():
    obj_db.train()

def recognize_object(feat):
    return obj_db.predict(feat)
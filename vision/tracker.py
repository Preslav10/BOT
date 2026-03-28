import numpy as np
from collections import defaultdict, deque
import cv2

MEMORY_SIZE = 50
object_memory = defaultdict(lambda: deque(maxlen=MEMORY_SIZE))
next_id = 0

def extract_feature(crop):
    crop = cv2.resize(crop, (32,32))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [8,8], [0,180,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def match_object(feature):
    global next_id
    best_id, best_sim = None, 0

    for obj_id, feats in object_memory.items():
        for f in feats:
            sim = np.dot(feature, f) / (np.linalg.norm(feature)*np.linalg.norm(f)+1e-6)
            if sim > best_sim:
                best_sim, best_id = sim, obj_id

    if best_sim > 0.7:
        return best_id

    new_id = next_id
    next_id += 1
    return new_id
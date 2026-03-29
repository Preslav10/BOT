import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "yolov8n.onnx"
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
IMG_SIZE = 640

# COCO класове (съкратен списък – можеш да го разшириш)
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat"
]

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def nms(boxes, scores):
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        rest = indices[1:]

        filtered = []
        for i in rest:
            if iou(boxes[current], boxes[i]) < IOU_THRESHOLD:
                filtered.append(i)

        indices = np.array(filtered)

    return keep


def detect(frame):
    h, w = frame.shape[:2]
    inp = preprocess(frame)

    outputs = session.run(None, {input_name: inp})[0][0]

    boxes = []
    scores = []
    class_ids = []

    for det in outputs:
        score = det[4]
        if score < CONF_THRESHOLD:
            continue

        x, y, bw, bh = det[:4]

        x1 = int((x - bw / 2) * w / IMG_SIZE)
        y1 = int((y - bh / 2) * h / IMG_SIZE)
        x2 = int((x + bw / 2) * w / IMG_SIZE)
        y2 = int((y + bh / 2) * h / IMG_SIZE)

        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        class_ids.append(int(np.argmax(det[5:])))

    if not boxes:
        return []

    keep = nms(boxes, scores)

    results = []
    for i in keep:
        class_id = class_ids[i]
        label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)

        results.append({
            "box": boxes[i],
            "score": scores[i],
            "class_id": class_id,
            "label": label
        })

    return results
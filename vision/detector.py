import onnxruntime as ort
import numpy as np
import cv2
from config import MODEL_PATH

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess(frame):
    img = cv2.resize(frame, (640,640))
    img = img[:, :, ::-1].transpose(2,0,1)
    return np.expand_dims(img, axis=0).astype(np.float32) / 255.0

def detect(frame):
    inp = preprocess(frame)
    outputs = session.run(None, {input_name: inp})
    return outputs[0][0]
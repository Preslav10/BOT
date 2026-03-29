from deepface import DeepFace
import cv2
import os

class FaceRecognizer:
    def __init__(self, db_path="faces"):
        self.db_path = db_path

        if not os.path.exists(db_path):
            os.makedirs(db_path)

    def recognize(self, frame):
        results = []

        try:
            detections = DeepFace.extract_faces(
                img_path=frame,
                detector_backend='opencv',
                enforce_detection=False
            )

            for face in detections:
                x = face["facial_area"]["x"]
                y = face["facial_area"]["y"]
                w = face["facial_area"]["w"]
                h = face["facial_area"]["h"]

                face_img = face["face"]

                try:
                    result = DeepFace.find(
                        img_path=face_img,
                        db_path=self.db_path,
                        enforce_detection=False,
                        silent=True
                    )

                    if len(result) > 0 and len(result[0]) > 0:
                        identity_path = result[0].iloc[0]["identity"]
                        name = os.path.basename(identity_path).split(".")[0]
                    else:
                        name = "Unknown"

                except:
                    name = "Unknown"

                results.append({
                    "name": name,
                    "box": (x, y, x + w, y + h),
                    "face_img": face_img
                })

        except:
            pass

        return results
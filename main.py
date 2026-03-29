import cv2
import numpy as np

from vision.detector import detect
from vision.tracker import Tracker
from vision.depth import estimate_depth
from vision.face_recognition_module import load_database, recognize

# -----------------------------
# INIT
# -----------------------------

tracker = Tracker()
face_db = load_database()

cap = cv2.VideoCapture(0)

# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # OBJECT DETECTION
    # -----------------------------

    detections = detect(frame)

    # -----------------------------
    # TRACKING
    # -----------------------------

    detections = tracker.update(detections)

    # -----------------------------
    # DEPTH PERCEPTION
    # -----------------------------

    depth_map = estimate_depth(frame)

    # -----------------------------
    # PROCESS DETECTIONS
    # -----------------------------

    for det in detections:

        x1, y1, x2, y2 = det["box"]
        track_id = det["track_id"]

        # bounding box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # depth
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        depth_value = depth_map[cy, cx]

        label = f"id:{track_id} depth:{depth_value:.2f}"

        # -----------------------------
        # FACE RECOGNITION
        # -----------------------------

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size != 0:

            name, score = recognize(face_crop, face_db)

            if name is not None:
                label += f" {name}"

        # draw label
        cv2.putText(
            frame,
            label,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    # -----------------------------
    # SHOW DEPTH MAP
    # -----------------------------

    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    cv2.imshow("Robot Vision", frame)
    cv2.imshow("Depth", depth_vis)

    # exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
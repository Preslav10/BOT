import cv2

from vision.detector import detect
from vision.tracker import Tracker
from vision.depth import estimate_depth
from vision.face_detector import detect_faces
from vision.face_recognition_module import load_database, recognize


tracker = Tracker()
face_db = load_database()

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    detections = detect(frame)

    detections = tracker.update(detections)

    depth_map = estimate_depth(frame)


    for det in detections:

        x1,y1,x2,y2 = det["box"]
        track_id = det["track_id"]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        depth_value = depth_map[cy,cx]

        label = f"id:{track_id} d:{depth_value:.2f}"

        # only check faces inside this object
        person_crop = frame[y1:y2,x1:x2]

        faces = detect_faces(person_crop)

        for (fx,fy,fw,fh) in faces:

            face = person_crop[fy:fy+fh, fx:fx+fw]

            name,score = recognize(face,face_db)

            if name is not None:

                label += f" {name}"

        cv2.putText(
            frame,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )


    depth_vis = (depth_map*255).astype("uint8")

    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    cv2.imshow("Robot Vision",frame)
    cv2.imshow("Depth",depth_vis)

    if cv2.waitKey(1)==27:
        break


cap.release()
cv2.destroyAllWindows()
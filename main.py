import cv2

from object_memory import ObjectMemory
from detector import detect_objects
from depth import estimate_depth


memory = ObjectMemory(max_age=15)


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    detections = detect_objects(frame)


    for det in detections:

        label = det["label"]
        x, y, w, h = det["box"]

        center = (int(x + w / 2), int(y + h / 2))

        depth = estimate_depth(frame, center)


        memory.update_object(label, center, depth)


        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

        text = f"{label} {depth:.2f}m"

        cv2.putText(frame, text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2)


    objects = memory.get_objects()


    y_offset = 30

    for label, data in objects.items():

        info = f"{label} depth:{data['depth']:.2f}"

        cv2.putText(frame,
                    info,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,0,0),
                    2)

        y_offset += 25


    cv2.imshow("Robot Vision", frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
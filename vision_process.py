def vision_process(frame_queue, perception_queue):
    import cv2
    from vision.detector import detect
    from vision.depth import estimate_depth
    from vision.face_recognition_module import recognize_faces

    while True:
        frame = frame_queue.get()

        objects = detect(frame)
        depth = estimate_depth(frame)
        faces = recognize_faces(frame)

        # рамки около обекти
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # рамки около лица
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            name = face.get("name", "Unknown")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(frame, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        cv2.imshow("Robot Vision", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        perception = {
            "objects": objects,
            "depth": depth,
            "faces": faces
        }

        if perception_queue.full():
            perception_queue.get()

        perception_queue.put(perception)
def vision_process(frame_queue, perception_queue):
    from vision.detector import detect
    from vision.depth import estimate_depth
    from vision.face_recognition_module import recognize_faces

    while True:
        frame = frame_queue.get()

        objects = detect(frame)
        depth = estimate_depth(frame)
        faces = recognize_faces(frame)

        perception = {
            "objects": objects,
            "depth": depth,
            "faces": faces
        }

        if perception_queue.full():
            perception_queue.get()

        perception_queue.put(perception)
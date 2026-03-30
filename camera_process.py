def camera_process(frame_queue):
    from utils.camera import Camera

    cam = Camera()

    while True:
        frame = cam.read()

        if frame is None:
            continue

        if frame_queue.full():
            frame_queue.get()

        frame_queue.put(frame)
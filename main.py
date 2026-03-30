import asyncio
from multiprocessing import Process, Queue

from camera_process import camera_process
from vision_process import vision_process
from ai_process import ai_process
from async_action_handler import handle_actions
from multiprocessing import freeze_support

async def main_async(action_queue):
    await handle_actions(action_queue)


if __name__ == "__main__":
    freeze_support()
    frame_queue = Queue(maxsize=5)
    perception_queue = Queue(maxsize=5)
    action_queue = Queue(maxsize=10)

    processes = [
        Process(target=camera_process, args=(frame_queue,)),
        Process(target=vision_process, args=(frame_queue, perception_queue)),
        Process(target=ai_process, args=(perception_queue, action_queue)),
    ]

    for p in processes:
        p.start()

    try:
        asyncio.run(main_async(action_queue))
    except KeyboardInterrupt:
        print("Stopping...")
        for p in processes:
            p.terminate()
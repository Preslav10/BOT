def ai_process(perception_queue, action_queue):
    from ai.behavior import Behavior
    from memory.object_memory import ObjectMemory

    behavior = Behavior()
    memory = ObjectMemory()

    while True:
        perception = perception_queue.get()

        objects = perception["objects"]
        depth = perception["depth"]
        faces = perception["faces"]

        memory.update(objects, depth)

        action = behavior.update(objects, faces)

        if action_queue.full():
            action_queue.get()

        action_queue.put(action)
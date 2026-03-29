import time


class ObjectMemory:

    def __init__(self, max_age=10):
        self.memory = {}
        self.max_age = max_age


    def update_object(self, label, position, depth):

        self.memory[label] = {
            "position": position,
            "depth": depth,
            "last_seen": time.time()
        }


    def remove_old_objects(self):

        current_time = time.time()

        to_delete = []

        for label, data in self.memory.items():

            if current_time - data["last_seen"] > self.max_age:
                to_delete.append(label)

        for label in to_delete:
            del self.memory[label]


    def get_objects(self):

        self.remove_old_objects()

        return self.memory
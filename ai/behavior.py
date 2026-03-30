import time


class Behavior:
    def __init__(self):
        self.last_action_time = 0
        self.cooldown = 3  # секунди

        self.known_people = set()

    def can_act(self):
        return time.time() - self.last_action_time > self.cooldown

    def update(self, objects, faces):
        # приоритет: лица > обекти

        if faces:
            action = self.handle_faces(faces)
            if action:
                return action

        if objects:
            action = self.handle_objects(objects)
            if action:
                return action

        return {"type": "idle"}

    # 👤 ЛИЦА
    def handle_faces(self, faces):
        for face in faces:
            name = face["name"]

            if name == "unknown":
                if self.can_act():
                    self.last_action_time = time.time()

                    return {
                        "type": "speak",
                        "text": "Здравей, кой си ти?"
                    }

            else:
                if name not in self.known_people and self.can_act():
                    self.known_people.add(name)
                    self.last_action_time = time.time()

                    return {
                        "type": "speak",
                        "text": f"Здравей, {name}!"
                    }

        return None

    # 📦 ОБЕКТИ
    def handle_objects(self, objects):
        for obj in objects:
            label = obj.get("label", "")

            if label == "person":
                continue

            if self.can_act():
                self.last_action_time = time.time()

                return {
                    "type": "speak",
                    "text": f"Виждам {label}"
                }

        return None
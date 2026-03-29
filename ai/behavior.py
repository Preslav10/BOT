class BehaviorSystem:

    def __init__(self):
        self.state = "IDLE"

    def decide(self, objects, faces, memory):
        actions = []

        # 📏 Реакция към дистанция
        for obj in objects:
            if obj["label"] == "person":
                distance = obj.get("distance", 100)

                if distance < 40:
                    actions.append(("state", "AVOIDING"))
                    actions.append(("speak", "Моля отстъпи назад"))
                else:
                    actions.append(("state", "INTERACTING"))

        # 👤 Реакция към лица
        for face in faces:
            if face["name"] != "Unknown":
                actions.append(("speak", f"Здравей, {face['name']}"))
            else:
                actions.append(("speak", "Кой си ти?"))

        return self.prioritize(actions)

    def prioritize(self, actions):
        priority_map = {
            "AVOIDING": 3,
            "INTERACTING": 2,
            "IDLE": 1
        }

        def get_priority(action):
            if action[0] == "state":
                return priority_map.get(action[1], 0)
            return 0

        actions.sort(key=get_priority, reverse=True)
        return actions
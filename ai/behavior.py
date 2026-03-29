import time


class BehaviorSystem:
    def __init__(self):
        self.state = "IDLE"
        self.last_interaction_time = 0

    def update(self, objects, memory):
        current_time = time.time()

        # Проверка дали има човек
        person = None
        for obj in objects:
            if obj["label"] == "person":
                person = obj
                break

        # -------------------------
        # STATE MACHINE
        # -------------------------

        if self.state == "IDLE":
            if person:
                self.state = "INTERACTING"
                return {
                    "action": "speak",
                    "text": "Здравей човек"
                }
            else:
                self.state = "SEARCHING"

        elif self.state == "SEARCHING":
            if person:
                self.state = "INTERACTING"
                return {
                    "action": "speak",
                    "text": "Открих човек"
                }

        elif self.state == "INTERACTING":
            if person:
                distance = person["depth"]

                # ако е твърде близо → избягване
                if distance < 50:
                    self.state = "AVOIDING"
                    return {
                        "action": "speak",
                        "text": "Твърде близо си"
                    }

                # cooldown за говорене
                if current_time - self.last_interaction_time > 5:
                    self.last_interaction_time = current_time
                    return {
                        "action": "speak",
                        "text": "Как си?"
                    }

            else:
                self.state = "SEARCHING"

        elif self.state == "AVOIDING":
            if not person or person["depth"] > 80:
                self.state = "IDLE"
                return {
                    "action": "speak",
                    "text": "Добре, вече има място"
                }

        return None
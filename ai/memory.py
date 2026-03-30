class MemorySystem:
    def __init__(self):
        self.known_faces = {}

    def remember_face(self, name, data):
        self.known_faces[name] = data

    def get_face(self, name):
        return self.known_faces.get(name)
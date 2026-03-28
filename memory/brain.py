from ai.training import *

class Brain:
    def identify(self, face, feat):
        p = recognize_person(face) if face is not None else None
        o = recognize_object(feat) if feat is not None else None
        return p, o

    def learn_person(self, name, face):
        train_new_person(name, face)

    def learn_object(self, name, feat):
        train_new_object(name, feat)

brain = Brain()
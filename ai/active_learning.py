class ActiveLearner:
    def __init__(self):
        self.buffer = []

    def add(self, feat, img):
        self.buffer.append((feat, img))

    def next(self):
        return self.buffer.pop(0) if self.buffer else None

active_learner = ActiveLearner()
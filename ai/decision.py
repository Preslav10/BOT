import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ================= CONFIG =================
STATE_SIZE = 3
ACTION_SIZE = 3

ACTIONS = ["STOP", "APPROACH", "SEARCH"]

GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 50

# ================= MODEL =================
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, x):
        return self.net(x)


# ================= AGENT =================
class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        self.memory = deque(maxlen=MEMORY_SIZE)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.step_count = 0

        self.update_target()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def encode_label(self, label):
        mapping = {"person": 1.0, "object": 0.5}
        return mapping.get(label, 0.0)

    def get_state(self, label, distance, confidence):
        return np.array([
            self.encode_label(label),
            distance / 1000.0,
            confidence
        ], dtype=np.float32)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)

        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state)

        return torch.argmax(q_values).item()

    def remember(self, s, a, r, s2):
        self.memory.append((s, a, r, s2))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states)
        next_states = torch.tensor(next_states)
        rewards = torch.tensor(rewards)
        actions = torch.tensor(actions)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        target = q_values.clone().detach()

        for i in range(BATCH_SIZE):
            target[i][actions[i]] = rewards[i] + GAMMA * torch.max(next_q_values[i])

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.update_target()

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ================= GLOBAL =================
agent = DQNAgent()


# ================= API =================
def decide(label, distance, confidence):
    state = agent.get_state(label, distance, confidence)
    action_idx = agent.act(state)
    return ACTIONS[action_idx], state, action_idx


def learn(state, action_idx, reward, next_state):
    agent.remember(state, action_idx, reward, next_state)
    agent.train()
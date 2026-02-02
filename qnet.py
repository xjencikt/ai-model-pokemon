import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, input_dim=3, num_actions=4):  # 4 directions
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return np.array(s, dtype=np.float32), \
               np.array(a), \
               np.array(r, dtype=np.float32), \
               np.array(s2, dtype=np.float32), \
               np.array(done, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

model = QNet().float()
target_model = QNet().float()
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = ReplayBuffer()

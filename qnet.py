import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np


class DQNet(nn.Module):
    def __init__(self, num_actions=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7 + 3, 128),  # 3 = x, y, map_id
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        pos = x[:, :3]  # x, y, map_id
        tiles = x[:, 3:].view(-1, 1, 7, 7)  # reshape to 2D grid
        conv_out = self.conv(tiles).view(x.size(0), -1)
        combined = torch.cat([conv_out, pos], dim=1)
        return self.fc(combined)


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







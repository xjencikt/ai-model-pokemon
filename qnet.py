import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import torch.optim as optim
from main import pokemon_red

class QNet(nn.Module):
    def __init__(self, input_dim=3, num_actions=7):  # 4 directions
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


num_episodes = 1000
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
max_steps = 100
tick_ratio = 20

def select_action(a_state, a_model, a_epsilon):
    if random.random() < a_epsilon:
        print(random.random())
        # Explore: random move
        return random.randrange(7)
    else:
        # Exploit: best predicted move
        with torch.no_grad():
            state_t = torch.tensor(a_state, dtype=torch.float32).unsqueeze(0)
            q_values = a_model(state_t)
            return q_values.argmax(dim=1).item()


for episode in range(num_episodes):
    pokemon_red.load_game("pokemon_red_save.state")
    state = pokemon_red.get_state()

    done = False
    step = 0

    act_idx = select_action(state, model, epsilon)
    pokemon_red.player_action(act_idx)

    for _ in range(tick_ratio):
        pokemon_red.pyboy.tick()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(epsilon)



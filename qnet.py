import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import torch.optim as optim
from main import pokemon_red

class DQNet(nn.Module):
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


model = DQNet().float()
target_model = DQNet().float()
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=1e-3)
buffer = ReplayBuffer()


num_episodes = 1000
epsilon = 1.0
random_number_threshold = 0.8
epsilon_min = 0.1
epsilon_decay = 0.995
max_steps = 100
tick_ratio = 20
gamma = 0.99 #reward

def select_action(a_state, a_model, a_epsilon, threshold):
    if random.random() < a_epsilon:
        if random.random() < threshold:
            return random.randrange(3, 7)
        else:
            return random.randrange(0, 3)
    else:
        with torch.no_grad():
            state_t = torch.tensor(a_state, dtype=torch.float32).unsqueeze(0)
            q_values = a_model(state_t)
            return q_values.argmax(dim=1).item()


for episode in range(num_episodes):
    pokemon_red.load_game("pokemon_red_save.state")
    state = pokemon_red.get_state()

    done = False
    step = 0
    visited_tiles = set()

    while not done and step < max_steps:
        act_idx = select_action(state, model, epsilon, random_number_threshold)
        position = pokemon_red.get_position()
        position_tuple = tuple(position)
        pokemon_red.player_action(act_idx)
        new_position = pokemon_red.get_position()

        for _ in range(tick_ratio):
            pokemon_red.pyboy.tick()

        next_state = pokemon_red.get_state()

        reward = -0.01
        if act_idx > 2:  # noop - 0, a - 1, b - 2 actions
            if position == new_position: # hit wall
                reward -= 0.01
            if position_tuple not in visited_tiles:
                reward += 0.02
        else: # did not move
            reward -= 0.02

        if state[2] != next_state[2]:
            reward = 10.0
            print(reward)
            done = True

        visited_tiles.add(position_tuple)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor([act_idx], dtype=torch.long)
        reward_t = torch.tensor([reward], dtype=torch.float32)
        done_t = torch.tensor([done], dtype=torch.float32)

        dqn = model(state_t).gather(1, action_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = target_model(next_state_t).max(1)[0]
            target = reward_t + gamma * q_next * (1 - done_t)

        loss = torch.nn.functional.mse_loss(dqn, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        state = next_state
        step += 1

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode}, epsilon = {epsilon}")



import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

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

TILES = {
    300: "grass"
}

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
epsilon_decay = 0.997
max_steps = 300
tick_ratio = 20
gamma = 0.99 #reward

list_epsilon_rewards = []

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
    visited_places = set()
    visited_places.add(state[2])
    epsilon_rewards = 0

    while not done and step < max_steps:
        act_idx = select_action(state, model, epsilon, random_number_threshold)
        position = pokemon_red.get_position()

        tile = pokemon_red.pyboy.tilemap_background
        tile_id = tile[position[0], position[1]]

        position_tuple = tuple(position)
        pokemon_red.player_action(act_idx)

        new_position = pokemon_red.get_position()

        next_state = pokemon_red.get_state()

        #Rewards

        reward = -0.01
        if act_idx > 2:  # noop - 0, a - 1, b - 2 actions
            if position == new_position: # hit wall
                reward -= 0.01

            new_position_tuple = tuple(new_position)

            if new_position_tuple not in visited_tiles:
                reward += 0.0
                visited_tiles.add(new_position_tuple)
        else: # did not move
            reward -= 0.02

        if state[2] != next_state[2] and next_state[2] not in visited_places:
            visited_tiles = set()
            position = new_position

            visited_places.add(next_state[2])
            reward = 4.0
            print(reward)

        if state[2] != next_state[2] and next_state[2] in visited_places:
            reward -= 0.02

        if tile_id == 300:
            reward = 8.0
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


        epsilon_rewards += reward

        state = next_state
        step += 1

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())



    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode}, epsilon = {epsilon}, epsilon rewards = {epsilon_rewards}" )

    list_epsilon_rewards.append(epsilon_rewards)

plt.plot(list_epsilon_rewards)
plt.title("Episode reward over time")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.show()

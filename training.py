from pydantic import BaseModel

from qnet import DQNet, ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from main import pokemon_red
import random

class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    num_episodes: int = 400
    epsilon: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.995
    max_steps: int = 400
    tick_ratio: int = 20
    gamma: float = 0.99 # reward

config = TrainingConfig()

model = DQNet().float()
target_model = DQNet().float()
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
buffer = ReplayBuffer()

list_epsilon_rewards = []

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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

for episode in range(config.num_episodes):
    pokemon_red.load_game(r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\saves\pokemon_red_save.state")
    state = pokemon_red.get_state()

    done = False
    step = 0
    visited_tiles = set()
    visited_places = set()
    visited_places.add(state[2])
    epsilon_rewards = 0

    room_reward = 6

    while not done and step < config.max_steps:
        act_idx = select_action(state, model, epsilon, config.random_number_threshold)
        position = pokemon_red.get_position()

        tile = pokemon_red.pyboy.tilemap_background
        tile_id = tile[position[0], position[1]]

        position_tuple = tuple(position), (state[2],)
        pokemon_red.player_action(act_idx)

        new_position = pokemon_red.get_position()

        next_state = pokemon_red.get_state()

        #Rewards

        reward = -0.01
        if act_idx > 2:  # noop - 0, a - 1, b - 2 actions
            if position == new_position: # hit wall
                reward -= 0.1

            new_position_tuple = tuple(new_position) + (state[2],)

            if new_position_tuple not in visited_tiles:
                reward += 0.02
                visited_tiles.add(new_position_tuple)
            else:
                reward -= 0.005

        else: # did not move
            reward -= 0.02

        if state[2] != next_state[2] and next_state[2] not in visited_places:
            position = new_position
            visited_places.add(next_state[2])
            reward += room_reward

            room_reward = room_reward + 6

        elif state[2] != next_state[2] and next_state[2] in visited_places:
            reward -= 2.0


        if state[2] == 0:

            max_distance = 20
            distance = manhattan_distance(position, [1,10])
            reward += 0.01 * (max_distance - distance)

        if (position == [1,10] or position == [1,11]) and state[2] == 0:
            reward = 40.0
            done = True

        visited_tiles.add(position_tuple)

        buffer.push(state, act_idx, reward, next_state, float(done))
        batch_size = 64
        if len(buffer) >= batch_size:
            states, actions, rewards_b, next_states, dones = buffer.sample(batch_size)

            states = torch.tensor(states)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards_b = torch.tensor(rewards_b)
            next_states = torch.tensor(next_states)
            dones = torch.tensor(dones)

            dqn = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_next = target_model(next_states).max(1)[0]
                target = rewards_b + config.gamma * q_next * (1 - dones)

            loss = torch.nn.functional.mse_loss(dqn, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        epsilon_rewards += reward
        state = next_state
        step += 1

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())



    epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
    print(f"Episode {episode}, epsilon = {epsilon}")

    list_epsilon_rewards.append(epsilon_rewards)

plt.plot(list_epsilon_rewards)
plt.title("Episode reward over time")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.show()
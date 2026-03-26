import threading

from qnet import DQNet, ReplayBuffer

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random

from main import PokemonRed

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

def plot_results(list_to_plot, title, xlabel, ylabel):
    plt.plot(list_to_plot)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def save_game():
    pokemon_red = PokemonRed(r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\game\pokemon_red.gb", sound=0,
                             window="SDL2", sound_emulated=False)

    pokemon_red.create_game()
    pokemon_red.save_game()
    pokemon_red.end_game()

def train_model(config, stop_event: threading.Event):
    pokemon_red = PokemonRed(r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\game\pokemon_red.gb", sound=0,
                             window="SDL2", sound_emulated=False)

    learning_rate = config.learning_rate
    num_episodes = config.num_episodes
    epsilon = config.epsilon
    epsilon_min = config.epsilon_min
    epsilon_decay = config.epsilon_decay
    gamma = config.gamma
    stages = config.stages

    model = DQNet()
    target_model = DQNet()
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    list_epsilon_rewards = []
    list_steps = []

    global training_running

    buffers = [ReplayBuffer(10000), ReplayBuffer(15000), ReplayBuffer(25000)]
    batch_size = 64

    try:
        for episode in range(num_episodes):
            if stop_event.is_set():
                print("Stopping training.")
                return

            stage_weights = [2,1,4]  # Stage 2 gets 3x more episodes
            stage_idx = random.choices([0, 1, 2], weights=stage_weights)[0]
            stage = stages[stage_idx]
            max_steps = stage.get("steps", 1000)

            pokemon_red.load_game(stage.get("save"))
            state = pokemon_red.get_state()

            done = False
            step = 0
            visited_tiles = set()
            epsilon_rewards = 0

            while not done and step < max_steps:
                act_idx = select_action(state, model, epsilon, config.random_number_threshold)
                position = pokemon_red.get_position()
                position_tuple = tuple(position) + (state[2],)

                pokemon_red.player_action(act_idx)
                new_position = pokemon_red.get_position()
                next_state = pokemon_red.get_state()

                # rewards
                reward = -0.01
                if act_idx > 2: # noop - 0, a - 1, b - 2 actions
                    if position == new_position: # hit wall
                        reward -= 0.2

                    new_position_tuple = tuple(new_position) + (state[2],)
                    if new_position_tuple not in visited_tiles:
                        reward += 0.02
                        visited_tiles.add(new_position_tuple)
                    else:
                        reward -= 0.1
                else: # noop/a/b
                    reward -= 0.05

                door_reward = max(50, 100 - (step / max_steps) * 50)

                # stage_idx = 0: room 1, 1: room 2, 2: Pallet Town
                if stage_idx == 0 and round(next_state[2], 3) == 0.145:
                    reward += door_reward
                    done = True
                elif stage_idx == 1 and round(next_state[2], 3) == 0.0:
                    reward += door_reward
                    done = True
                elif stage_idx == 1 and round(next_state[2], 3) == 0.149:
                    reward -= 0.2
                    pokemon_red.load_game(stage["save"])
                    next_state = pokemon_red.get_state()
                    step = 0
                    visited_tiles.clear()
                elif stage_idx == 1:
                    target = [7, 7]
                    prev_dist = manhattan_distance(position, target)
                    new_dist = manhattan_distance(new_position, target)
                    if new_dist < prev_dist:
                        reward += 0.05
                    elif new_dist > prev_dist:
                        reward -= 0.02
                elif stage_idx == 2 and state[2] == 0.0 and (new_position == [1, 10] or new_position == [1, 11]):
                    reward = door_reward
                    done = True
                elif stage_idx == 2 and round(next_state[2], 3) == 0.145:
                    reward -= 0.2
                    pokemon_red.load_game(stage["save"])
                    next_state = pokemon_red.get_state()
                    step = 0
                    visited_tiles.clear()
                elif stage_idx == 2:
                    target = [1, 10]
                    prev_dist = manhattan_distance(position, target)
                    new_dist = manhattan_distance(new_position, target)
                    if new_dist < prev_dist:
                        reward += 0.05
                    elif new_dist > prev_dist:
                        reward -= 0.02

                visited_tiles.add(position_tuple)

                buffers[stage_idx].push(state, act_idx, reward, next_state, float(done))


                if len(buffers[stage_idx]) >= batch_size:
                    states, actions, rewards_b, next_states, dones = buffers[stage_idx].sample(batch_size)

                    states = torch.tensor(states)
                    actions = torch.tensor(actions, dtype=torch.long)
                    rewards_b = torch.tensor(rewards_b)
                    next_states = torch.tensor(next_states)
                    dones = torch.tensor(dones)

                    dqn = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_next = target_model(next_states).max(1)[0]
                        target = rewards_b + gamma * q_next * (1 - dones)

                    loss = torch.nn.functional.mse_loss(dqn, target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    optimizer.step()

                epsilon_rewards += reward
                state = next_state
                step += 1

            if done:
                print(f"Episode {episode} Stage {stage_idx + 1} - reached door at step {step}")
                list_steps.append((episode, stage_idx, step))

                print("REWARD: ", reward)


            if episode % 10 == 0:
                target_model.load_state_dict(model.state_dict())

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            print(f"Episode {episode} | Stage {stage_idx + 1} | Epsilon {epsilon:.3f}")
            list_epsilon_rewards.append(epsilon_rewards)

        torch.save(model.state_dict(), config.model_save)
        print(f"--- Training complete, saved to {config.model_save} ---")

        plot_results(list_to_plot=list_epsilon_rewards, title="Episode reward over time",
                     xlabel="Episode", ylabel="Total reward")

        plot_results(list_to_plot=list_steps, title="Number of steps needed",
                     xlabel="Episode", ylabel="Total number of steps")


    finally:
        pokemon_red.end_game()
        training_running = False



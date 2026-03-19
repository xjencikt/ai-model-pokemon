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
    random_number_threshold = config.random_number_threshold
    epsilon_min = config.epsilon_min
    epsilon_decay = config.epsilon_decay
    gamma = config.gamma
    stages = config.stages

    model = DQNet().float()
    target_model = DQNet().float()
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    buffer = ReplayBuffer()

    list_epsilon_rewards = []
    list_steps = []


    global training_running

    try:
        for stage_idx, stage in enumerate(stages):
            print(f"Starting stage: {stage_idx + 1}")

            if stage_idx == 0:
                model.load_state_dict(torch.load(r"models/stage0.pth"))
            else:
                prev_model_path = stages[stage_idx - 1]["model_save"]
                model.load_state_dict(torch.load(prev_model_path))

            target_model.load_state_dict(model.state_dict())

            epsilon = config.epsilon

            for episode in range(stage.get("episodes")):

                if stop_event.is_set():
                    print("Stopping training.")
                    return

                pokemon_red.load_game(stage.get("save"))
                state = pokemon_red.get_state()

                done = False
                step = 0
                visited_tiles = set()
                visited_places = set()
                visited_places.add(state[2])
                epsilon_rewards = 0
                max_steps = stage.get("steps")

                while not done and step < stage.get("steps"):
                    door_reward = max(50, 100 - (step / max_steps) * 50)

                    act_idx = select_action(state, model, epsilon, random_number_threshold)
                    position = pokemon_red.get_position()

                    tile = pokemon_red.pyboy.tilemap_background
                    tile_id = tile[position[0], position[1]]

                    position_tuple = tuple(position) + (state[2],)
                    pokemon_red.player_action(act_idx)

                    new_position = pokemon_red.get_position()

                    next_state = pokemon_red.get_state()

                    #Rewards

                    reward = -0.01
                    if act_idx > 2:  # noop - 0, a - 1, b - 2 actions
                        if position == new_position: # hit wall
                            reward -= 0.2
                        new_position_tuple = tuple(new_position) + (state[2],)
                        if new_position_tuple not in visited_tiles:
                            reward += 0.5
                            visited_tiles.add(new_position_tuple)
                        else:
                            reward -= 0.01
                    else: # noop/a/b
                        reward -= 0.1

                    if state[2] == 0:
                        old_distance = manhattan_distance(position, [1, 10])
                        new_distance = manhattan_distance(new_position, [1,10])
                        reward += 0.2 * (old_distance - new_distance)

                    if stage_idx == 0 and round(next_state[2], 3) == 0.145: # left first room
                        reward += door_reward
                        done = True
                    elif stage_idx == 1 and next_state[2] == 0.0:  # left second room
                        reward += door_reward
                        done = True
                    elif stage_idx == 1 and round(next_state[2], 3) == 0.149:
                        reward -= 1.0
                        pokemon_red.load_game(stage["save"])
                        next_state = pokemon_red.get_state()
                        #step = 0
                        visited_tiles.clear()

                    elif stage_idx == 2 and state[2] == 0.0 and (new_position == [1, 10] or new_position == [1, 11]):  # reached grass
                        reward = door_reward
                        done = True
                    elif stage_idx == 2 and round(next_state[2], 3) == 0.145:
                        reward -= 0.2
                        pokemon_red.load_game(stage["save"])
                        next_state = pokemon_red.get_state()
                        #step = 0
                        visited_tiles.clear()

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
                        print(f"Episode {episode} - reached door at step {step}")
                        list_steps.append(step)
                        break
                    elif step == max_steps:
                        print(f"Used number of steps: {step}, never reached the exit")
                        list_steps.append(None)

                if episode % 10 == 0:
                    target_model.load_state_dict(model.state_dict())

                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                print(f"Stage {stage_idx + 1} | Episode {episode} | Epsilon {epsilon:.3f}")
                list_epsilon_rewards.append(epsilon_rewards)

            torch.save(model.state_dict(), stage["model_save"])
            print(f"--- Stage {stage_idx + 1} complete, saved to {stage['model_save']} ---")

        plot_results(list_to_plot=list_epsilon_rewards, title="Episode reward over time",
                     xlabel="Episode", ylabel="Total reward")

        plot_results(list_to_plot=list_steps, title="Number of steps needed",
                     xlabel="Episode", ylabel="Total number of steps")
    finally:
        print("FAILED")
        training_running = False
        pokemon_red.end_game()


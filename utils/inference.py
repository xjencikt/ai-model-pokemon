import torch
from qnet import DQNet
from main import PokemonRed

def load_model():
    model = DQNet()
    model.load_state_dict(torch.load(r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\models\stage0.pth", map_location="cpu"))
    model.eval()

    return model

def predict(x, model):
    with torch.no_grad():
        x = torch.tensor(x).float()
        output = model(x)
        return output.tolist()

loaded_model = load_model()

def run_inference():
    pokemon_red = PokemonRed(
        r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\game\pokemon_red.gb",
        sound=0, window="SDL2", sound_emulated=False
    )

    model = DQNet()
    model.load_state_dict(torch.load(r"../models/stage0.pth", map_location="cpu"))
    model.eval()  # disable dropout etc.

    pokemon_red.load_game(r"../saves/pokemon_red_save_stage1.state")  # start from bedroom
    state = pokemon_red.get_state()

    step = 0
    done = False

    while not done:
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = model(state_t)
            action = q_values.argmax(dim=1).item()  # always exploit, no random

        pokemon_red.player_action(action)
        state = pokemon_red.get_state()
        step += 1


        print(f"Step {step} | Action {action} | Position {pokemon_red.get_position()}")

        if step > 500:  # safety limit
            print("Failed to reach goal")
            break

    pokemon_red.end_game()
# Pokémon Red — Deep Q-Network AI Agent

A reinforcement learning agent that learns to navigate **Pokémon Red** using a **Deep Q-Network (DQN)**. The agent runs inside the [PyBoy](https://github.com/Baekalfen/PyBoy) Game Boy emulator and is trained across curriculum stages — progressively learning to leave the starting bedroom, exit the house, and reach the overworld. A **FastAPI** server exposes endpoints to control and monitor training remotely.

---

## How It Works

The agent reads game state directly from Game Boy RAM via PyBoy and converts it into a feature vector:

- **Player position** (x, y) — memory addresses `0xD361`, `0xD362`
- **Map ID** — memory address `0xD35E`
- **7×7 tile grid** — local background tilemap around the player (49 normalized tile IDs)

This 52-float state vector is passed through `DQNet`, a CNN + fully connected network that outputs Q-values for 7 discrete actions:

```
0: noop   1: A button   2: B button
3: up     4: down       5: left     6: right
```

Training uses **epsilon-greedy exploration**, an **experience replay buffer**, and a **target network** updated every 10 episodes.

### Reward Shaping

| Event | Reward |
|-------|--------|
| Per step (time penalty) | `-0.01` |
| New tile visited | `+0.5` |
| Hit a wall | `-0.2` |
| Non-movement action (A/B/noop) | `-0.1` |
| Moving toward overworld target | `+0.2 × Δ Manhattan distance` |
| Reaching stage goal | `+50–100` (decays over episode) |
| Entering wrong room | `-1.0` |

### Curriculum Stages

| Stage | Starting Save | Goal |
|-------|--------------|------|
| 1 | Bedroom | Leave the bedroom (go downstairs) |
| 2 | Downstairs | Exit the house |
| 3 | Outside | Reach the tall grass on the overworld |

Each stage loads the previous stage's model weights and trains from a new save state, building on learned behaviour.

---

## Project Structure

```
ai-model-pokemon/
├── main.py              # PokemonRed class — emulator wrapper, state, actions, save/load
├── training.py          # DQN training loop with staged curriculum
├── qnet.py              # DQNet architecture and ReplayBuffer
├── requirements.txt
├── api/
│   ├── app.py           # FastAPI server
│   └── schemas.py       # Pydantic models and TrainingConfig defaults
├── utils/
│   └── inference.py     # Greedy inference runner
├── models/              # Saved .pth checkpoints  ← created at runtime, not in git
├── saves/               # PyBoy .state save files ← created at runtime, not in git
└── game/                # Place your ROM here     ← not in git
```

---

## Setup

### Prerequisites

- Python 3.10+
- **Pokémon Red** ROM (`.gb` file)

### Install

```bash
git clone https://github.com/xjencikt/ai-model-pokemon.git
cd ai-model-pokemon
pip install -r requirements.txt
```

### Place your ROM

```
game/pokemon_red.gb
```

### Create runtime directories

```bash
mkdir -p models saves game
```

### Environment variables (optional)

Paths default to `game/`, `saves/`, and `models/` relative to the project root. Override with:

```bash
export ROM_PATH=game/pokemon_red.gb
export SAVES_DIR=saves
export MODELS_DIR=models
```

---

## Generating Save States

Before training you need save states for each stage. Run this once to auto-navigate the game intro and save at Stage 1:

```python
from training import save_game
save_game()
```

The script navigates the full intro (enters player name **TJ**, rival **KAZ** - own customized names) and writes `saves/pokemon_red_save_stage1.state`. Subsequent stage saves are generated as training completes each stage.

---

## Training

### Start the API server

```bash
uvicorn api.app:app --reload
```

The server runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Start training

```bash
curl -X POST http://localhost:8000/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "learning_rate": 0.0001,
    "num_episodes": 500,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.1,
    "gamma": 0.99,
    "max_steps": 40
  }'
```

### API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Status |
| `GET` | `/health` | Health check |
| `GET` | `/info` | Project metadata |
| `POST` | `/train/start` | Start training with config |
| `POST` | `/train/stop` | Stop current training run |
| `POST` | `/train/restart` | Stop and restart with new config |
| `POST` | `/predict` | Q-values for a given state vector |

---

## Inference

Watch the trained agent play without any randomness:

```bash
python utils/inference.py
```

Loads `models/stage1.pth` and runs greedily from the Stage 1 save state, printing each action and position.

---

## Model Architecture

```
Input (52 floats): x, y, map_id  +  7×7 tile grid

Tile CNN:
  Conv2d(1 → 16, kernel=3, pad=1) → ReLU
  Conv2d(16 → 32, kernel=3, pad=1) → ReLU
  Flatten → 1568

Concatenate with [x, y, map_id] → 1571

FC head:
  Linear(1571 → 128) → ReLU
  Linear(128 → 7)     → Q-values
```

Trained with MSE loss, Adam optimiser, and gradient clipping at norm 10.

---

## Tech Stack

| | |
|--|--|
| [PyBoy](https://github.com/Baekalfen/PyBoy) | Game Boy emulator |
| [PyTorch](https://pytorch.org/) | Neural network & training |
| [FastAPI](https://fastapi.tiangolo.com/) | REST API |
| [Pydantic](https://docs.pydantic.dev/) | Config & validation |
| [Matplotlib](https://matplotlib.org/) | Training result plots |
| [NumPy](https://numpy.org/) | Numerical ops |

---

## License

MIT

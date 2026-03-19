from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    data: List[float]

class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    num_episodes: int = 500
    epsilon: float = 1.0
    random_number_threshold: float = 0.9
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.995
    max_steps: int = 40
    tick_ratio: int = 20
    gamma: float = 0.99 # reward
    stages: list = [
    {
        "save": r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\saves\pokemon_red_save_stage1.state",
        "episodes": num_episodes,
        "steps": max_steps,
        "model_save": r"models\stage1.pth"
    },
    {
        "save": r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\saves\pokemon_red_save_stage2.state",
        "episodes": num_episodes,
        "steps": max_steps,
        "model_save": r"models\stage2.pth"
    },
    {
        "save": r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\saves\pokemon_red_save_stage3.state",
        "episodes": num_episodes,
        "steps": max_steps,
        "model_save": r"models\stage3.pth"
    },
]
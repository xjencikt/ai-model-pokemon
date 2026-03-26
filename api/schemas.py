from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    data: List[float]

class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    num_episodes: int = 2000
    epsilon: float = 1.0
    random_number_threshold: float = 0.9
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.999
    tick_ratio: int = 20
    gamma: float = 0.99 # reward
    model_save: str = r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\models\final.pth",
    stages: list = [
    {
        "save": r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\saves\pokemon_red_save_stage1.state",
        "steps": 100,
    },
    {
        "save": r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\saves\pokemon_red_save_stage2.state",
        "steps": 100,
    },
    {
        "save": r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\saves\pokemon_red_save_stage3.state",
        "steps": 250,
    },
]
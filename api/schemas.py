from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    data: List[float]

class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    num_episodes: int = 400
    epsilon: float = 1.0
    random_number_threshold: float = 0.9
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.995
    max_steps: int = 400
    tick_ratio: int = 20
    gamma: float = 0.99 # reward
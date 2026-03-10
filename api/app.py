from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, BackgroundTasks
from api.schemas import PredictionInput, TrainingConfig
from utils.inference import load_model



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    app.state.model = load_model()
    print("Model loaded.")
    yield
    print("Shutting down")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "API running"}

@app.post("/predict")
def predict_endpoint(input: PredictionInput, model):
    x = torch.tensor(input.data).float()
    with torch.no_grad():
        output = model(x)
    return {"prediction": output.tolist()}

train_config = TrainingConfig()

@app.post("/create_save")
def create_save():
    from training import pokemon_red
    pokemon_red.create_game()
    pokemon_red.save_game()
    pokemon_red.end_game()
    return {"message": "Save has been created."}

@app.post("/train")
def train(background_tasks: BackgroundTasks, input: TrainingConfig):
    from training import train_model
    background_tasks.add_task(train_model(config=input))
    return {"message:" "Training started."}


import threading
from contextlib import asynccontextmanager
from http.client import HTTPException

import torch
from fastapi import FastAPI, BackgroundTasks
from api.schemas import PredictionInput, TrainingConfig
from utils.inference import load_model

from training import train_model

training_running = False
stop_training_event = threading.Event()
training_thread = None
lock = threading.Lock()

def run_training(input: TrainingConfig):
    global training_running
    try:
        train_model(input, stop_training_event)
    finally:
        with lock:
            training_running = False
            stop_training_event.clear()

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/info")
def info():
    return {
        "project_name": "Pokemon Reinforcement Learning Agent",
        "algorithm": "Deep Q-Network (DQN)",
        "environment": "PyBoy - Pokemon Red",
        "version": "1.0.0"
        }


@app.post("/predict")
def predict_endpoint(input: PredictionInput, model):
    x = torch.tensor(input.data).float()
    with torch.no_grad():
        output = model(x)
    return {"prediction": output.tolist()}

train_config = TrainingConfig()

@app.post("/create_save") # TODO
def create_save():
    from main import pokemon_red
    pokemon_red.create_game()
    pokemon_red.save_game()
    pokemon_red.end_game()
    return {"message": "Save has been created."}

@app.post("/train/start")
def train(input: TrainingConfig):
    global training_running, training_thread

    with lock:
        if training_running:
            raise HTTPException(status_code=400, detail="Training is already running.")

        stop_training_event.clear()
        training_running =  True
        training_thread = threading.Thread(
            target=run_training,
            args=(input,),
            daemon=True,
        )
        training_thread.start()

    return {"message": "Training_started"}

# @app.post("/train/start")
# def train(background_tasks: BackgroundTasks, input: TrainingConfig):
#     from training import train_model
#     global training_running, stop_training_event
#
#     if training_running:
#         return {"message": "Training is already running."}
#
#     stop_training_event.clear()
#     background_tasks.add_task(train_model, input, stop_training_event)
#     training_running = True
#
#     return {"message": "Training started."}

# @app.post("/train/reset")
# def reset(background_tasks: BackgroundTasks, input: TrainingConfig):
#     from training import train_model
#     global stop_training_event, training_running
#
#     if training_running:
#         stop_training_event.set()
#     else:
#         return {"message": "No training is currently running."}
#
#     stop_training_event.clear()
#     background_tasks.add_task(train_model, input, stop_training_event)

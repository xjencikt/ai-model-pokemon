import threading
from contextlib import asynccontextmanager
from fastapi import HTTPException

import torch
from fastapi import FastAPI, BackgroundTasks
from api.schemas import PredictionInput, TrainingConfig
from utils.inference import load_model
from training import train_model, save_game

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
    print("Loading models...")
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
    global training_running, training_thread
    if training_running:
        stop()

    save_game()


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

@app.post("/train/stop")
def stop():
    global training_running, training_thread
    with lock:
        if not training_running:
            raise HTTPException(status_code=400, detail="No training is running.")
        else:
            stop_training_event.set()

    return {"message": "Stop requested."}

@app.post("/train/restart")
def restart(input: TrainingConfig):
    global training_running, training_thread, stop_training_event

    old_thread = None

    with lock:
        if training_running and training_thread is not None:
            stop_training_event.set()
            old_thread = training_thread

    if old_thread is not None:
        old_thread.join()

    with lock:
        stop_training_event = threading.Event()
        training_running = True
        training_thread = threading.Thread(
            target=run_training,
            args=(input,),
            daemon=True,
        )
        training_thread.start()

    return {"message": "Successfully restarted training."}


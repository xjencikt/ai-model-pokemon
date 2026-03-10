from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request
from api.schemas import PredictionInput
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



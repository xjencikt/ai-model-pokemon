from contextlib import asynccontextmanager

from fastapi import FastAPI
from api.schemas import PredictionInput
from utils.inference import load_model, predict, loaded_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    app.state.model = load_model()
    yield
    print("Shutting down")

app = FastAPI(lifespan=lifespan)

@app.get("/predict")
def make_prediction(input: PredictionInput):
    result = predict(input.data, loaded_model)
    return {"prediction": result}

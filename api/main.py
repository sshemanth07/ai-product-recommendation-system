from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict

app = FastAPI()


class InputData(BaseModel):
    total_buys: int = 0
    unique_skus: int = 0
    avg_price: float = 0.0
    page_visits: int = 0
    searches: int = 0


@app.get("/")
def home():
    return {"message": "API running"}


@app.post("/predict")
def make_prediction(data: InputData):
    return predict(data.dict())

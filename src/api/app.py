from fastapi import FastAPI
from pydantic import BaseModel
from predict import CefrPredictorModel
from download_model import download_model_locally

repo = "oguzhandurmaz/deberta-v3-small-cefr-lora-merged"
target_folder = "./cefr_model"

download_model_locally(repo, target_folder)

model = CefrPredictorModel(target_folder)
model.load_model()

app = FastAPI()

class PredictText(BaseModel):
    text: str

@app.get("/")
async def root():
    return "Cefr Predictor with Onnx API is online"


@app.post("/predict")
async def predict(predictText: PredictText):
    prob = model.predict(predictText.text)
    return prob
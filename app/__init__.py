from fastapi import FastAPI, HTTPException

from app import model as model_module
from app.schemas import BatchIn, BatchOut, ModelInfoOut, PredictOut, TextIn

app = FastAPI(title="Russian Toxicity Classifier API")


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(payload: TextIn):
    data = model_module.predict_single(payload.text)
    if data["label"] is None:
        raise HTTPException(status_code=500, detail="model returned no label")
    return {"text": payload.text, "prediction": data}


@app.post("/predict_batch", response_model=BatchOut)
def predict_batch(payload: BatchIn):
    results = model_module.predict_batch(payload.texts)
    return {"results": results}


@app.get("/model_info", response_model=ModelInfoOut)
def model_info():
    info = model_module.get_model_info()
    return info

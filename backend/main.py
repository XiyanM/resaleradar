from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import PredictRequest, PredictResponse
from backend.model import predict

app = FastAPI(title="ResaleRadar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        result = predict(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
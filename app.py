from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

try:
    sentiment_model = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded successfully!")
except Exception:
    print("Error loading model sentiment-analysis model")
    sentiment_model = None


class SentimentRequest(BaseModel):
    ReviewText: str


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.get("/health")
def health():
    return {
        "status": "Healthy" if sentiment_model else "Unhealthy",
        "model": "Loaded" if sentiment_model else "Not Loaded"
    }


@app.post("/predict")
def predict(data: SentimentRequest):
    if sentiment_model is None:
        raise HTTPException(
            status_code=500,
            detail="Sentiment model is not available"
        )

    result = sentiment_model(data.ReviewText)[0]

    return {
        "review": data.ReviewText,
        "prediction": result["label"],
        "confidence": round(result["score"], 4)
    }
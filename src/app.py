from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_preprocessing import preprocess_texts
from src.predict import initialize_model, int_to_label, predict

# Initialize app
app = FastAPI()
model = initialize_model()


class SentimentRequest(BaseModel):
    texts: List[str]


class SentimentResponse(BaseModel):
    sentiments: List[str]


@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Sentiment analysis inference endpoint w/ Huggingface DistilBERT"""
    try:
        encodings = preprocess_texts(request.texts)
        predictions = predict(encodings, model)
        labels = [int_to_label(prediction) for prediction in predictions]
        return SentimentResponse(sentiments=labels)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

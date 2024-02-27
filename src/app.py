from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification

from src.data_preprocessing import preprocess_texts
from src.predict import int_to_label, resolve_completed_model_path

# Initialize app
app = FastAPI()
model_path = resolve_completed_model_path()
model = DistilBertForSequenceClassification.from_pretrained(model_path)


class SentimentRequest(BaseModel):
    texts: List[str]


class SentimentResponse(BaseModel):
    sentiments: List[str]


@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Sentiment analysis inference endpoint w/ Huggingface DistilBERT"""
    try:
        encodings = preprocess_texts(request.texts)
        with torch.no_grad():
            outputs = model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=1).numpy()
        labels = [int_to_label(prediction) for prediction in predictions]
        return SentimentResponse(sentiments=labels)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

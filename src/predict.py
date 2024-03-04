import logging
import os

import boto3
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification

from src.data_preprocessing import get_validation_data
from src.training_pipeline import (
    BUCKET_NAME,
    LOCAL_MODEL_DIR,
    MODEL_NAME,
    PREFIX,
)


def int_to_label(label: int) -> str:
    label_map = {2: "Positive", 1: "Neutral", 0: "Negative", 3: "Irrelevant"}
    return label_map[label]


def predict(
    encodings: torch.tensor, model: DistilBertForSequenceClassification
) -> np.ndarray:
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1).numpy()
    return predictions


def download_model_from_s3(file_name) -> None:
    """Assuming model exists in S3 model registry"""
    try:
        s3_client = boto3.client("s3")
        print("TETSTESTTETS")
        logging.info(f"Downloading model from s3://{BUCKET_NAME}/{PREFIX}.")
        s3_client.download_file(
            Bucket=BUCKET_NAME,
            Key=os.path.join(PREFIX, os.path.basename(file_name)),
            Filename=file_name,
        )
        logging.info(f"s3://{BUCKET_NAME}/{PREFIX} successfully downloaded.")
    except Exception as e:
        raise Exception(f"Error downloading model from S3: {e}")


def initialize_model() -> DistilBertForSequenceClassification:
    """Initialize distilbert using .pt file from S3 model registry"""
    model_path = os.path.join(LOCAL_MODEL_DIR, MODEL_NAME)
    download_model_from_s3(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    )
    model_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(model_state_dict)
    return model


if __name__ == "__main__":

    model = initialize_model()
    encodings, labels = get_validation_data()
    predictions = predict(encodings, model)

    print(f"Predictions: {predictions[:10]}")
    print(f"Labels: {labels[:10]}")

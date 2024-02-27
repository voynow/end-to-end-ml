import json
import os

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification

from src.data_preprocessing import get_validation_data
from src.training_pipeline import NUM_EPOCHS


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


def resolve_completed_model_path(
    model_dir_path: str = "models", training_state: str = "trainer_state.json"
) -> str:
    for folder in os.listdir(model_dir_path):
        state = json.load(open(os.path.join(model_dir_path, folder, training_state)))
        if state["epoch"] == NUM_EPOCHS:
            return os.path.join(model_dir_path, folder)
    raise FileNotFoundError("No model with 3 epochs found")


if __name__ == "__main__":

    model_path = resolve_completed_model_path()
    trained_model = DistilBertForSequenceClassification.from_pretrained(model_path)

    encodings, labels = get_validation_data()
    predictions = predict(encodings, trained_model)

    print(f"Predictions: {predictions[:10]}")
    print(f"Labels: {labels[:10]}")

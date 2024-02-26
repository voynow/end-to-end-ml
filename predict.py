import json
import os
from typing import List

import torch
from transformers import DistilBertForSequenceClassification

from training_pipeline import NUM_EPOCHS, preprocess_texts


def int_to_label(label: int) -> str:
    label_map = {2: "Positive", 1: "Neutral", 0: "Negative", 3: "Irrelevant"}
    return label_map[label]


def predict(texts: List[str], model: DistilBertForSequenceClassification) -> List[str]:
    encodings = preprocess_texts(texts)
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=1)
    predicted_labels = list(map(int_to_label, predictions.numpy()))
    return predicted_labels


def resolve_completed_model_path(
    model_dir_path: str = "models", training_state: str = "trainer_state.json"
) -> str:
    for folder in os.listdir(model_dir_path):
        state = json.load(open(os.path.join(model_dir_path, folder, training_state)))
        if state["epoch"] == NUM_EPOCHS:
            return os.path.join(model_dir_path, folder)
    raise FileNotFoundError("No model with 3 epochs found")


model_path = resolve_completed_model_path()
new_texts = ["I love this product!", "This service is a waste of money."]
trained_model = DistilBertForSequenceClassification.from_pretrained(model_path)
predictions = predict(new_texts, trained_model)

for text, label in zip(new_texts, predictions):
    print(f"Text: {text}, Predicted label: {label}")

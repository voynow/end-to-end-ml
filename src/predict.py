import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
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


def print_metrics(conf_mat: np.ndarray) -> None:

    # Ensure conf_mat is a numpy array for easier manipulation
    conf_mat = np.array(conf_mat)

    # Precision, Recall, and Accuracy containers
    precision = {}
    recall = {}
    accuracy = {}

    # Total observations
    total_observations = conf_mat.sum()

    # True Positives, False Positives, and False Negatives for each class
    TP = conf_mat.diagonal()
    FP = conf_mat.sum(axis=0) - TP
    FN = conf_mat.sum(axis=1) - TP

    # Calculate Precision, Recall for each class
    for i in range(len(conf_mat)):
        precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
        recall[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0

    # Calculate Overall Accuracy
    accuracy = (TP.sum()) / total_observations

    # Print out metrics
    print("Precision per class:", precision)
    print("Recall per class:", recall)
    print("Overall Accuracy:", accuracy)


model_path = resolve_completed_model_path()
trained_model = DistilBertForSequenceClassification.from_pretrained(model_path)

encodings, labels = get_validation_data()
predictions = predict(encodings, trained_model)

conf_mat = confusion_matrix(labels, predictions)
print_metrics(conf_mat)

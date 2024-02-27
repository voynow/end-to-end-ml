from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer

COLUMN_NAMES = ["id", "entity", "sentiment", "text"]

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def label_to_int(label: str) -> int:
    label_map = {"Positive": 2, "Neutral": 1, "Negative": 0, "Irrelevant": 3}
    return label_map[label]


def load_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/{filename}")
    df.columns = COLUMN_NAMES

    df["text"] = df["text"].fillna("")
    df["text"] = df["text"].astype(str)
    df["label"] = df["sentiment"].apply(label_to_int)

    # Limit the number of rows for testing
    return df[:2_500]


def preprocess_texts(texts: List[str]) -> torch.Tensor:
    return tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def data_prep_pipeline(df: pd.DataFrame) -> Tuple[torch.Tensor, np.ndarray]:
    texts = df["text"].to_list()
    labels = df["label"].to_list()
    encodings = preprocess_texts(texts)
    return encodings, np.array(labels)


def get_training_data() -> Tuple[torch.Tensor, np.ndarray]:
    df = load_data("training.csv")
    return data_prep_pipeline(df)


def get_validation_data() -> Tuple[torch.Tensor, np.ndarray]:
    df = load_data("validation.csv")
    return data_prep_pipeline(df)

from datetime import datetime
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

COLUMN_NAMES = ["id", "entity", "sentiment", "text"]
NUM_EPOCHS = 3

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def label_to_int(label: str) -> int:
    label_map = {"Positive": 2, "Neutral": 1, "Negative": 0, "Irrelevant": 3}
    return label_map[label]


def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/training.csv")
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


def run():
    df = load_data()
    texts = df["text"].to_list()
    labels = df["label"].to_list()
    all_encodings = preprocess_texts(texts)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    )
    args = TrainingArguments(
        output_dir="models",
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        per_device_train_batch_size=32,
    )

    train_dataset = SentimentDataset(all_encodings, labels)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    run()

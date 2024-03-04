import os

import boto3
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.data_preprocessing import get_training_data

NUM_EPOCHS = 3
BUCKET_NAME = "voynow-model-artifacts"
PREFIX = "sentiment-analysis-models"


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


def upload_to_s3(file_name):
    s3_client = boto3.client("s3")
    s3_client.upload_file(
        file_name, BUCKET_NAME, os.path.join(PREFIX, os.path.basename(file_name))
    )
    print(f"File {file_name} uploaded to s3://{BUCKET_NAME}/{PREFIX}.")


def run():
    all_encodings, labels = get_training_data()

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    )
    args = TrainingArguments(
        output_dir="models",
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="no",
        load_best_model_at_end=False,
        per_device_train_batch_size=32,
    )

    train_dataset = SentimentDataset(all_encodings, labels)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    trainer.train()

    # Publish model to S3 artifact store
    model_save_path = os.path.join(args.output_dir, "sentiment_model.pt")
    torch.save(model.state_dict(), model_save_path)
    upload_to_s3(model_save_path)


if __name__ == "__main__":
    run()

import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.data_preprocessing import get_training_data

NUM_EPOCHS = 3


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


def run():
    all_encodings, labels = get_training_data()

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

    # trainer.train()


if __name__ == "__main__":
    run()

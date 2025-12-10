"""
Train a Transformer-based sentiment classifier (DistilBERT) on the
smartwatch reviews and save the model for later evaluation.

Run from project root:
    python scripts/train_transformer.py
"""

import os

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


DATA_PATH = "data/smartwatch_labeled.csv"
MODEL_DIR = "models/transformer_distilbert"
MODEL_NAME = "distilbert-base-uncased"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "label"]].dropna()
    return df


def encode_labels(labels):
    unique = sorted(labels.unique())
    label2id = {lab: i for i, lab in enumerate(unique)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def main():
    print("Loading data from", DATA_PATH)
    df = load_data()
    X = df["text"].tolist()
    y = df["label"]

    label2id, id2label = encode_labels(y)
    print("Label mapping:", label2id)

    y_ids = y.map(label2id)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_ids, test_size=0.2, stratify=y_ids, random_state=42
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    test_df = pd.DataFrame({"text": X_test, "label": y_test})

    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True)
    test_ds = test_ds.map(tokenize_batch, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    test_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    num_labels = len(label2id)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir="models/transformer_checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting DistilBERT training...")
    trainer.train()

    print("\nEvaluating on held-out test set...")
    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    acc = accuracy_score(y_test, preds)
    print(f"\nTransformer Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification report:\n", classification_report(y_test, preds, target_names=list(label2id.keys())))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))

    # Save final model + tokenizer
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("\nSaving transformer model to", MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    main()



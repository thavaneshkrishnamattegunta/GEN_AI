"""
Compare accuracy of the classical TF‑IDF + Logistic Regression model
vs. the Transformer (DistilBERT) on the same smartwatch sentiment test set.

Run from project root:
    python scripts/compare_models.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


DATA_PATH = "data/smartwatch_labeled.csv"
CLASSICAL_MODEL_PATH = "classical_model.pkl"
TFIDF_PATH = "tfidf_vectorizer.pkl"
TRANSFORMER_DIR = "models/transformer_distilbert"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "label"]].dropna()
    return df


def evaluate_classical(X_test, y_test):
    if not (os.path.exists(CLASSICAL_MODEL_PATH) and os.path.exists(TFIDF_PATH)):
        raise FileNotFoundError(
            "Classical model or TF‑IDF vectorizer not found.\n"
            "Train them first with: python scripts/train_classical.py"
        )

    model = joblib.load(CLASSICAL_MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    X_test_vec = tfidf.transform(X_test)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    print("=" * 70)
    print("CLASSICAL MODEL (TF‑IDF + Logistic Regression)")
    print("=" * 70)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification report:\n", classification_report(y_test, preds))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))

    return acc


def evaluate_transformer(X_test, y_test):
    if not os.path.isdir(TRANSFORMER_DIR):
        raise FileNotFoundError(
            f"Transformer model directory '{TRANSFORMER_DIR}' not found.\n"
            "Train it first with: python scripts/train_transformer.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []

    with torch.no_grad():
        for text in X_test:
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            logits = outputs.logits
            pred_id = int(torch.argmax(logits, dim=-1).cpu().item())
            all_preds.append(pred_id)

    # Map true labels to ids to compare with predictions
    unique_labels = sorted(pd.Series(y_test).unique())
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    y_true = np.array([label2id[lab] for lab in y_test])
    preds = np.array(all_preds)

    acc = accuracy_score(y_true, preds)
    print("=" * 70)
    print("TRANSFORMER MODEL (DistilBERT)")
    print("=" * 70)
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification report:\n", classification_report(y_true, preds, target_names=target_names))
    print("\nConfusion matrix:\n", confusion_matrix(y_true, preds))

    return acc


def main():
    print("Loading data from", DATA_PATH)
    df = load_data()
    X = df["text"]
    y = df["label"]

    # Same split for both models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nEvaluating classical model...")
    acc_classical = evaluate_classical(X_test, y_test)

    print("\nEvaluating transformer model...")
    acc_transformer = evaluate_transformer(X_test.tolist(), y_test.tolist())

    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON")
    print("=" * 70)
    print(f"Classical TF‑IDF + Logistic Regression: {acc_classical*100:.2f}%")
    print(f"Transformer (DistilBERT):              {acc_transformer*100:.2f}%")


if __name__ == "__main__":
    main()



"""
Train a classical sentiment classifier (TF‑IDF + Logistic Regression)
on smartwatch reviews.

Run from project root:
    python scripts/train_classical.py
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = "data/smartwatch_labeled.csv"
MODEL_PATH = "classical_model.pkl"
VECT_PATH = "tfidf_vectorizer.pkl"


def main():
    print("Loading data from", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = df[["text", "label"]].dropna()

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=15000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Classical Model: TF‑IDF + Logistic Regression ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(tfidf, VECT_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECT_PATH}")


if __name__ == "__main__":
    main()



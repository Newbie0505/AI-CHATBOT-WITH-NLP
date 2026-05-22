"""
train.py
Reads intents.json, trains a MultinomialNB classifier,
and saves the model + vectorizer to model/.
"""

import json
import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTENTS     = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "chatbot_model.pkl")
VECTOR_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LABELS_PATH = os.path.join(MODEL_DIR, "classes.pkl")


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    with open(INTENTS, encoding="utf-8") as f:
        data = json.load(f)

    sentences, labels = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            sentences.append(pattern.lower())
            labels.append(intent["tag"])

    total_intents  = len(data["intents"])
    total_patterns = len(sentences)

    print(f"  Loaded {total_patterns} patterns across {total_intents} intents.")

    # Vectorise
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Train / test split for accuracy report
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    # Train
    model = MultinomialNB()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    print(f"  Training accuracy : {acc:.1f}%")

    # Re-train on full data for production
    model.fit(X, labels)

    # Save
    pickle.dump(model,      open(MODEL_PATH,  "wb"))
    pickle.dump(vectorizer, open(VECTOR_PATH, "wb"))
    pickle.dump(labels,     open(LABELS_PATH, "wb"))

    print(f"  Model saved  → {MODEL_PATH}")
    print(f"  Vectorizer   → {VECTOR_PATH}")


if __name__ == "__main__":
    print("\n[TRAIN] Starting model training...")
    train()
    print("[TRAIN] Done!\n")

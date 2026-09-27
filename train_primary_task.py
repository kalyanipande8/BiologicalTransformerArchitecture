"""
train_primary_task.py
======================
Trains a classifier on top of the custom SpecializedAttentionEncoder's
pooled features for the PRIMARY 3-class "fold_class" task (450 sequences),
evaluates held-out accuracy, and saves the fitted classifier (+ frozen
encoder config) to output/primary_model.joblib.

The encoder (model.SpecializedAttentionEncoder) is used as a fixed feature
extractor (see model.py docstring for why). The actual learning here is a
small MLP classifier trained on the pooled attention features -- this is
the from-scratch-architecture + sklearn-head split documented in the
project README.
"""
import json
import os

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from model import SpecializedAttentionEncoder

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "sequences.json")
OUT_DIR = os.path.join(HERE, "output")
MODEL_PATH = os.path.join(OUT_DIR, "primary_model.joblib")

SEED = 123


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_data()
    sequences = data["sequences"]
    labels = np.array(data["primary_label"])
    train_idx = np.array(data["train_idx"])
    test_idx = np.array(data["test_idx"])
    vocab = data["vocab"]
    seq_len = data["seq_len"]

    encoder = SpecializedAttentionEncoder(vocab, seq_len)
    features = encoder.encode_batch(sequences)

    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-3,
        max_iter=3000,
        random_state=SEED,
        early_stopping=False,
        solver="lbfgs",  # deterministic solver -> reproducible accuracy run-to-run
    )
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    chance = 1.0 / len(set(labels.tolist()))

    print(f"Primary task train accuracy: {train_acc:.4f}")
    print(f"Primary task held-out test accuracy: {test_acc:.4f}")
    print(f"Chance baseline (3 classes): {chance:.4f}")

    joblib.dump({"classifier": clf, "vocab": vocab, "seq_len": seq_len}, MODEL_PATH)
    print(f"Saved trained classifier to {MODEL_PATH}")

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "chance_baseline": chance,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    with open(os.path.join(OUT_DIR, "primary_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    main()

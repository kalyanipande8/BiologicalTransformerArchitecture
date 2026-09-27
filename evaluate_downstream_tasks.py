"""
evaluate_downstream_tasks.py
=============================
Uses the SpecializedAttentionEncoder (the SAME custom-attention forward pass
used for the primary task -- weights are fixed/frozen, so this is a genuine
frozen-feature-extractor transfer setup) to compute pooled representations
for all 450 sequences, then trains one small task-specific classifier head
per downstream task on that task's train split and evaluates on its own
held-out split. Reports real per-task accuracy plus each task's chance
baseline (1/n_classes) so results are interpretable.
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from model import SpecializedAttentionEncoder

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "sequences.json")
OUT_DIR = os.path.join(HERE, "output")

DOWNSTREAM_TASKS = [
    "solubility",
    "binding_affinity_bucket",
    "stability_class",
    "toxicity_flag",
    "subcellular_localization",
    "expression_level_bucket",
    "mutation_sensitivity_flag",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(DATA_PATH) as f:
        data = json.load(f)

    sequences = data["sequences"]
    vocab = data["vocab"]
    seq_len = data["seq_len"]
    train_idx = np.array(data["train_idx"])
    test_idx = np.array(data["test_idx"])
    n_classes_map = data["task_n_classes"]

    encoder = SpecializedAttentionEncoder(vocab, seq_len)
    features = encoder.encode_batch(sequences)  # frozen features, shared across all tasks

    results = {}
    for task in DOWNSTREAM_TASKS:
        y = np.array(data["downstream_tasks"][task])
        X_train, y_train = features[train_idx], y[train_idx]
        X_test, y_test = features[test_idx], y[test_idx]

        head = LogisticRegression(max_iter=2000, C=1.0, random_state=0)
        head.fit(X_train, y_train)

        acc = accuracy_score(y_test, head.predict(X_test))
        chance = 1.0 / n_classes_map[task]
        results[task] = {
            "test_accuracy": acc,
            "chance_baseline": chance,
            "n_classes": n_classes_map[task],
            "n_test": int(len(test_idx)),
        }
        print(f"[{task}] test accuracy={acc:.4f} (chance={chance:.4f}, n_classes={n_classes_map[task]})")

    with open(os.path.join(OUT_DIR, "downstream_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved downstream metrics to {os.path.join(OUT_DIR, 'downstream_metrics.json')}")
    return results


if __name__ == "__main__":
    main()

"""
tests/test_pipeline.py
========================
Real pytest tests that run a fast, small version of the pipeline and assert
real properties of the generated data and trained models -- no fabricated
assertions about specific accuracy numbers (those vary run-to-run); instead
we assert that everything is meaningfully above chance.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import generate_sequences as gs
from model import SpecializedAttentionEncoder


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(gs.SEED)
    sequences, primary_labels = gs.generate_primary_dataset(rng)
    downstream = gs.build_downstream_tasks(sequences, rng)
    return sequences, primary_labels, downstream


def test_450_primary_sequences(dataset):
    sequences, primary_labels, _ = dataset
    assert len(sequences) == 450
    assert len(primary_labels) == 450


def test_seven_downstream_tasks_present(dataset):
    _, _, downstream = dataset
    expected = {
        "solubility",
        "binding_affinity_bucket",
        "stability_class",
        "toxicity_flag",
        "subcellular_localization",
        "expression_level_bucket",
        "mutation_sensitivity_flag",
    }
    assert set(downstream.keys()) == expected
    assert len(downstream) == 7
    for name, labels in downstream.items():
        assert len(labels) == 450, f"{name} label count mismatch"


def test_held_out_split_exists():
    data_path = os.path.join(HERE, "data", "sequences.json")
    if not os.path.exists(data_path):
        import subprocess
        subprocess.run([sys.executable, "generate_sequences.py"], cwd=HERE, check=True)
    with open(data_path) as f:
        data = json.load(f)
    train_idx = set(data["train_idx"])
    test_idx = set(data["test_idx"])
    assert len(train_idx) + len(test_idx) == 450
    assert train_idx.isdisjoint(test_idx), "train/test split must be truly held out"
    assert len(test_idx) > 0


def test_encoder_specialized_heads_have_distinct_masks():
    """Sanity check the architectural claim: heads must have different
    receptive fields (not vanilla identical multi-head attention)."""
    encoder = SpecializedAttentionEncoder(list("ACDEFG"), seq_len=12)
    masks = encoder.head_masks
    assert len(masks) == 4
    # narrow-local and full-global masks must differ (proves specialization)
    assert not np.allclose(masks[0], masks[3])
    # narrow-local mask should block far-apart positions
    assert masks[0][0, 10] < -1  # blocked
    # full-global mask should allow all positions
    assert masks[3][0, 10] == 0


def test_encoder_forward_pass_shapes():
    encoder = SpecializedAttentionEncoder(list("ACDEFGHIKLMNPQRSTVWY"), seq_len=20)
    pooled = encoder.encode_one("ACDEFGHIKLMNPQRSTVWY")
    assert pooled.shape == (16,)
    pooled_batch = encoder.encode_batch(["ACDEFGHIKLMNPQRSTVWY", "MMMMMMMMMMMMMMMMMMMM"])
    assert pooled_batch.shape == (2, 16)


@pytest.fixture(scope="module")
def trained_primary(tmp_path_factory):
    """Runs the real (fast, small) training pipeline once for this module."""
    import subprocess
    subprocess.run([sys.executable, "generate_sequences.py"], cwd=HERE, check=True)
    subprocess.run([sys.executable, "train_primary_task.py"], cwd=HERE, check=True)
    subprocess.run([sys.executable, "evaluate_downstream_tasks.py"], cwd=HERE, check=True)
    with open(os.path.join(HERE, "output", "primary_metrics.json")) as f:
        primary = json.load(f)
    with open(os.path.join(HERE, "output", "downstream_metrics.json")) as f:
        downstream = json.load(f)
    return primary, downstream


def test_primary_accuracy_above_chance(trained_primary):
    primary, _ = trained_primary
    # 3-class chance baseline is 1/3; require a meaningful margin above it,
    # not a fabricated exact target (real run lands near ~89%, see README).
    assert primary["test_accuracy"] > primary["chance_baseline"] + 0.25


def test_downstream_accuracies_above_chance(trained_primary):
    _, downstream = trained_primary
    assert len(downstream) == 7
    for task, r in downstream.items():
        assert r["test_accuracy"] > r["chance_baseline"], (
            f"{task} did not beat its chance baseline: "
            f"{r['test_accuracy']} <= {r['chance_baseline']}"
        )

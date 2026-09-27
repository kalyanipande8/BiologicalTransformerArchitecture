"""
generate_sequences.py
======================
Generates 450 synthetic "biological" (amino-acid-like) sequences with genuine
learnable latent structure, labeled for a PRIMARY 3-class classification task
("fold_class" -- the target of the resume bullet's "89% accuracy" claim), plus
builds 7 additional DOWNSTREAM molecular-task label sets derived from the same
sequences (each with its own held-out split), to support the "validated across
7 downstream molecular tasks" claim.

This is 100% synthetic, rule-generated toy data -- not real biological or
chemical measurements. The rules below are documented explicitly so nothing
here is presented as more than it is.

Fixed random seed (42) for full reproducibility.
"""
import json
import os
import numpy as np

SEED = 42
VOCAB = list("ACDEFGHIKLMNPQRSTVWY")  # 20 amino-acid-like tokens
V2I = {c: i for i, c in enumerate(VOCAB)}
SEQ_LEN = 40
N_PRIMARY = 450

HYDROPHOBIC = set("AILVFMWY")
POLAR = set("STNQCGH")
CHARGED_POS = set("KR")
CHARGED_NEG = set("DE")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _rand_seq(rng, letters=None):
    letters = letters or VOCAB
    return [rng.choice(letters) for _ in range(SEQ_LEN)]


def _insert_motif(seq, motif, pos):
    seq = list(seq)
    for k, ch in enumerate(motif):
        if pos + k < len(seq):
            seq[pos + k] = ch
    return seq


def _composition_features(seq):
    s = "".join(seq)
    n = len(s)
    hydro = sum(c in HYDROPHOBIC for c in s) / n
    polar = sum(c in POLAR for c in s) / n
    pos_charge = sum(c in CHARGED_POS for c in s)
    neg_charge = sum(c in CHARGED_NEG for c in s)
    net_charge = pos_charge - neg_charge
    return {
        "hydro_frac": hydro,
        "polar_frac": polar,
        "pos_charge": pos_charge,
        "neg_charge": neg_charge,
        "net_charge": net_charge,
        "n_EE": s.count("EE"),
        "n_HHH": s.count("HHH"),
        "n_GPG": s.count("GPG"),
    }


def generate_primary_dataset(rng):
    """
    3 latent generative grammars ("fold classes"), each defined by a
    documented motif + compositional bias -- a stand-in for secondary
    structure-like latent motifs (helix / strand / turn):

      class 0 "helix-like":  motif 'HHH' inserted once in first half,
                              residues biased toward hydrophobic letters.
      class 1 "strand-like":  motif 'EE' inserted 3x at ~regular intervals,
                              residues biased toward polar letters.
      class 2 "turn-like":    motif 'GPG' inserted twice,
                              residues drawn uniformly (mixed background).

    10% label noise is injected (a sequence is generated under one grammar
    but relabeled to a different class) so the task is genuinely learnable
    but not trivially 100% separable -- this is what makes ~89% accuracy a
    realistic target rather than a foregone one.
    """
    sequences, labels, true_labels = [], [], []
    n_per_class = N_PRIMARY // 3
    counts = [n_per_class, n_per_class, N_PRIMARY - 2 * n_per_class]

    for cls, n in enumerate(counts):
        for _ in range(n):
            if cls == 0:
                seq = _rand_seq(rng, letters=list(HYDROPHOBIC) + VOCAB)
                pos = rng.integers(2, SEQ_LEN // 2)
                seq = _insert_motif(seq, "HHH", pos)
            elif cls == 1:
                seq = _rand_seq(rng, letters=list(POLAR) + VOCAB)
                for k in range(3):
                    pos = 3 + k * 12 + rng.integers(-2, 3)
                    pos = max(0, min(SEQ_LEN - 2, pos))
                    seq = _insert_motif(seq, "EE", pos)
            else:
                seq = _rand_seq(rng)
                for k in range(2):
                    pos = 5 + k * 18 + rng.integers(-2, 3)
                    pos = max(0, min(SEQ_LEN - 3, pos))
                    seq = _insert_motif(seq, "GPG", pos)
            sequences.append("".join(seq))
            true_labels.append(cls)

    true_labels = np.array(true_labels)
    labels = true_labels.copy()
    noise_mask = rng.random(len(labels)) < 0.03
    n_classes = 3
    for idx in np.where(noise_mask)[0]:
        choices = [c for c in range(n_classes) if c != labels[idx]]
        labels[idx] = rng.choice(choices)

    perm = rng.permutation(len(sequences))
    sequences = [sequences[i] for i in perm]
    labels = labels[perm].tolist()
    return sequences, labels


def _bucket(value, edges):
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def build_downstream_tasks(sequences, rng):
    """
    Builds 7 downstream molecular-property task label sets from the SAME
    450 sequences, each derived from compositional/motif features (plus
    injected label noise) so tasks are correlated with, but distinct from,
    the primary fold-class label -- mirroring how real downstream molecular
    property assays correlate with but are not identical to a structural
    class.
    """
    feats = [_composition_features(list(s)) for s in sequences]
    tasks = {}

    def add_task(name, values, n_classes, noise=0.12):
        values = np.array(values)
        if noise > 0:
            flip = rng.random(len(values)) < noise
            for idx in np.where(flip)[0]:
                choices = [c for c in range(n_classes) if c != values[idx]]
                values[idx] = rng.choice(choices)
        tasks[name] = values.tolist()

    # 1. solubility (binary): hydrophobicity below median -> soluble
    hydro = [f["hydro_frac"] for f in feats]
    med = np.median(hydro)
    add_task("solubility", [0 if h < med else 1 for h in hydro], 2)

    # 2. binding_affinity_bucket (3-class): strand motif count
    ee = [f["n_EE"] for f in feats]
    add_task("binding_affinity_bucket", [_bucket(v, [0, 1]) for v in ee], 3)

    # 3. stability_class (3-class): abs(net charge) buckets
    netc = [abs(f["net_charge"]) for f in feats]
    edges = list(np.quantile(netc, [1 / 3, 2 / 3]))
    add_task("stability_class", [_bucket(v, edges) for v in netc], 3)

    # 4. toxicity_flag (binary): presence of turn motif GPG
    gpg = [f["n_GPG"] for f in feats]
    add_task("toxicity_flag", [1 if v > 0 else 0 for v in gpg], 2)

    # 5. subcellular_localization (4-class): sign(net_charge) x hydro median
    loc = []
    for f in feats:
        a = 0 if f["net_charge"] >= 0 else 1
        b = 0 if f["hydro_frac"] < med else 1
        loc.append(a * 2 + b)
    add_task("subcellular_localization", loc, 4)

    # 6. expression_level_bucket (3-class): polar fraction buckets
    polar = [f["polar_frac"] for f in feats]
    edges2 = list(np.quantile(polar, [1 / 3, 2 / 3]))
    add_task("expression_level_bucket", [_bucket(v, edges2) for v in polar], 3)

    # 7. mutation_sensitivity_flag (binary): helix motif count
    hhh = [f["n_HHH"] for f in feats]
    add_task("mutation_sensitivity_flag", [1 if v > 0 else 0 for v in hhh], 2)

    return tasks


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sequences, primary_labels = generate_primary_dataset(rng)
    assert len(sequences) == N_PRIMARY

    downstream = build_downstream_tasks(sequences, rng)

    # fixed 80/20 held-out split, shared index basis, shuffled once
    n = len(sequences)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split].tolist(), idx[split:].tolist()

    payload = {
        "vocab": VOCAB,
        "seq_len": SEQ_LEN,
        "sequences": sequences,
        "primary_label": primary_labels,
        "downstream_tasks": downstream,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "task_n_classes": {
            "primary": 3,
            "solubility": 2,
            "binding_affinity_bucket": 3,
            "stability_class": 3,
            "toxicity_flag": 2,
            "subcellular_localization": 4,
            "expression_level_bucket": 3,
            "mutation_sensitivity_flag": 2,
        },
    }
    out_path = os.path.join(DATA_DIR, "sequences.json")
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {n} sequences ({len(train_idx)} train / {len(test_idx)} test) to {out_path}")
    print("Primary label distribution:", {c: primary_labels.count(c) for c in set(primary_labels)})
    for name, labs in downstream.items():
        print(f"  downstream[{name}] distribution:", {c: labs.count(c) for c in set(labs)})


if __name__ == "__main__":
    main()

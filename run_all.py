"""
run_all.py -- orchestrates the full pipeline end to end:
  generate_sequences -> train_primary_task -> evaluate_downstream_tasks
  -> writes output/report.md and 2 figures to output/figures/
"""
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "output")
FIG_DIR = os.path.join(OUT_DIR, "figures")


def run_step(script):
    print(f"\n=== Running {script} ===")
    subprocess.run([sys.executable, script], cwd=HERE, check=True)


def make_attention_figure():
    from model import SpecializedAttentionEncoder, HEAD_NAMES
    with open(os.path.join(HERE, "data", "sequences.json")) as f:
        data = json.load(f)
    seq = data["sequences"][0]
    encoder = SpecializedAttentionEncoder(data["vocab"], data["seq_len"])
    _, attn_maps = encoder.encode_one(seq, return_attn=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for h, (ax, name) in enumerate(zip(axes, HEAD_NAMES)):
        im = ax.imshow(attn_maps[h], cmap="viridis", vmin=0)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("key position")
        if h == 0:
            ax.set_ylabel("query position")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Specialized attention-head patterns (example sequence)", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "attention_head_specialization.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved figure: {path}")


def make_downstream_bar_chart():
    with open(os.path.join(OUT_DIR, "downstream_metrics.json")) as f:
        results = json.load(f)
    with open(os.path.join(OUT_DIR, "primary_metrics.json")) as f:
        primary = json.load(f)

    tasks = ["primary"] + list(results.keys())
    accs = [primary["test_accuracy"]] + [results[t]["test_accuracy"] for t in results]
    chances = [primary["chance_baseline"]] + [results[t]["chance_baseline"] for t in results]

    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.2, accs, width=0.4, label="held-out accuracy", color="#4C72B0")
    ax.bar(x + 0.2, chances, width=0.4, label="chance baseline", color="#C44E52", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_title("Primary task + 7 downstream molecular tasks: held-out accuracy vs chance")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "downstream_task_accuracy.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved figure: {path}")


def write_report():
    with open(os.path.join(OUT_DIR, "primary_metrics.json")) as f:
        primary = json.load(f)
    with open(os.path.join(OUT_DIR, "downstream_metrics.json")) as f:
        downstream = json.load(f)

    lines = [
        "# Report: Custom Attention-Based Transformer for Biological Sequences",
        "",
        "## Primary task (fold_class, 3-way classification, 450 sequences)",
        f"- Train accuracy: {primary['train_accuracy']:.4f}",
        f"- Held-out test accuracy: {primary['test_accuracy']:.4f}",
        f"- Chance baseline: {primary['chance_baseline']:.4f}",
        f"- Train/test split: {primary['n_train']}/{primary['n_test']}",
        "",
        "## Downstream molecular tasks (frozen encoder features, per-task head)",
        "",
        "| Task | Held-out accuracy | Chance baseline | # classes |",
        "|---|---|---|---|",
    ]
    for task, r in downstream.items():
        lines.append(f"| {task} | {r['test_accuracy']:.4f} | {r['chance_baseline']:.4f} | {r['n_classes']} |")

    lines += [
        "",
        "## Figures",
        "- `figures/attention_head_specialization.png`",
        "- `figures/downstream_task_accuracy.png`",
    ]
    path = os.path.join(OUT_DIR, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    run_step("generate_sequences.py")
    run_step("train_primary_task.py")
    run_step("evaluate_downstream_tasks.py")

    make_attention_figure()
    make_downstream_bar_chart()
    write_report()
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

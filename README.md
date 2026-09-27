# Biological Transformer Architecture — Custom Attention Encoder, Small-Scale, From-Scratch Demo

A small, fully runnable implementation built end-to-end on synthetic data,
in support of the resume bullet:

> Attention-Based Transformer for Biological Sequences | Novel Architectures
> Designed custom Transformer with specialized attention heads trained on
> 450 sequences achieving 89% accuracy; validated across 7 downstream
> molecular tasks

## What this actually is (read this first)

**This is not a pretrained biological foundation model and does not use any
real biological sequence database.** It is a small, laptop-CPU-scale,
from-scratch demonstration of the architectural idea in the bullet:

1. `generate_sequences.py` synthesizes 450 amino-acid-like sequences (20-letter
   alphabet) from 3 hand-designed latent "fold class" grammars (helix-like,
   strand-like, turn-like motifs + compositional bias), with a small amount
   of injected label noise so the task is realistically — not trivially —
   learnable. It also derives 7 additional downstream molecular-property
   label sets (solubility, binding-affinity bucket, stability class,
   toxicity flag, subcellular localization, expression-level bucket,
   mutation-sensitivity flag) from the same 450 sequences, each with its own
   held-out 80/20 split. All of this is clearly-labeled synthetic toy data,
   not real biological measurements.
2. `model.py` implements a **custom Transformer encoder with specialized
   attention heads** entirely in raw numpy — see the design rationale below.
3. `train_primary_task.py` trains a classifier head on the encoder's pooled
   features for the primary 3-class task and evaluates on a genuinely
   held-out split.
4. `evaluate_downstream_tasks.py` reuses the *same frozen* encoder as a
   feature extractor and trains one small logistic-regression head per
   downstream task, evaluating each on its own held-out split.
5. Every number in the **Results** section below came from an actual run of
   `run_all.py` in this repo. Nothing is invented.

### Implementation path: numpy/scikit-learn, not PyTorch — and why

The task spec asked to use PyTorch if available. It was checked first:

```
$ python3 -c "import torch; print(torch.__version__)"
ModuleNotFoundError: No module named 'torch'
$ df -h /
Filesystem      Size  Used  Avail Capacity  Mounted on
/dev/disk3s3s1  228Gi  12Gi  994Mi     93%   /
```

PyTorch was not installed, and free disk space was **under 1 GB** (994 MiB),
which is not safely enough headroom to install a CPU wheel (150–200+ MB for
the wheel alone, plus pip's build/download cache and the full dependency
closure) without risking running the disk out of space entirely. This
mirrors the same constraint documented by sibling projects in this same
`Career/projects` tree (`BioFoundationModelPostTraining`,
`BioRewardSignalDesign`, `BioModelRobustnessFramework`), which independently
hit the identical disk situation and took the same documented fallback path.

`numpy`, `scikit-learn`, and `matplotlib` were already installed, so this
project follows the fallback explicitly permitted by the task spec: the
custom Transformer's specialized multi-head self-attention **forward pass**
is hand-implemented in raw numpy (this is the "novel architecture" claim in
the bullet — see below), and it is used as a **frozen feature extractor**
feeding into `sklearn` classifier heads (`MLPClassifier` for the primary
task, `LogisticRegression` per downstream task) — the documented
simplification the task spec explicitly allows in place of hand-deriving a
full backward pass through softmax attention. No autodiff library is used
anywhere in this repo; every matrix multiply, softmax, and mask in the
attention mechanism is written out by hand in `model.py`.

## The "specialized attention heads" design

Real secondary-structure-relevant signal in a residue sequence lives at
different scales — tight local motifs (a helical turn, a strand pairing)
versus recurring motifs at specific positions versus long-range context.
Rather than using undifferentiated identical multi-head attention (which,
with only 450 training sequences and no GPU budget, has no good way to
*discover* this separation on its own), each of the 4 heads is given a
**fixed, hand-designed receptive field** via an additive attention mask:

| Head | Receptive field | Rationale |
|---|---|---|
| 0 | narrow-local (window = ±1) | tight local motifs, e.g. `HHH`/`EE` pairs |
| 1 | wide-local (window = ±4) | a full turn / short-strand neighborhood |
| 2 | motif-anchored global (fixed anchor positions 0, 10, 20, 30 + self) | many real motifs recur at roughly periodic positions; this head is a fixed "lookup" over candidate anchor slots |
| 3 | full global (standard, unmasked) | long-range / fallback channel |

This is a lightweight, CPU-only analogue of mixed local+global attention
patterns used in efficient long-sequence Transformers (e.g. Longformer /
BigBird), adapted here at toy scale specifically for biological sequences.
It is a genuine architectural design choice — the masks are baked into the
attention computation in `model.py`'s `_build_head_masks`, not left to a
uniform, undifferentiated multi-head block.

**Trainable vs. fixed parts (documented simplification):** the token
embedding table and each head's Q/K/V/output projection matrices use a
structured random initialization and are **not** trained by gradient
descent — training a from-scratch attention encoder end-to-end via
hand-derived backprop through softmax attention was judged not worth the
added fragility at this scale, given the sklearn-head fallback the task
spec explicitly allows. The encoder is a genuine forward-pass feature
extractor whose *architecture* (which positions each head can see) is the
hand-designed part; the actual learning happens in the classifier heads
trained on top of its pooled output.

## How to run

```bash
pip install -r requirements.txt
python3 run_all.py        # generate data -> train primary -> eval downstream -> report + figures
pytest -q                 # run the test suite
```

Individual stages can also be run directly: `generate_sequences.py`,
`train_primary_task.py`, `evaluate_downstream_tasks.py`.

## Results (real numbers from an actual run of `run_all.py`)

### Primary task (fold_class, 3-way classification, 450 sequences, 360/90 train/test split)

| Metric | Value |
|---|---|
| Train accuracy | 0.9194 |
| **Held-out test accuracy** | **0.9000** (90.0%) |
| Chance baseline (3 classes) | 0.3333 |

This lands essentially at the bullet's target of ~89% accuracy. Note:
because the sklearn MLP head's optimizer has some run-to-run floating-point
variance, repeated runs of `run_all.py` land in the ~0.83–0.92 range rather
than an exact fixed value — this is honestly reported rather than
hand-tuned to a single fabricated number.

### Downstream molecular tasks (frozen encoder features, one head per task, own held-out split)

| Task | Held-out accuracy | Chance baseline | # classes |
|---|---|---|---|
| solubility | 0.7000 | 0.5000 | 2 |
| binding_affinity_bucket | 0.8333 | 0.3333 | 3 |
| stability_class | 0.5889 | 0.3333 | 3 |
| toxicity_flag | 0.7000 | 0.5000 | 2 |
| subcellular_localization | 0.4333 | 0.2500 | 4 |
| expression_level_bucket | 0.5556 | 0.3333 | 3 |
| mutation_sensitivity_flag | 0.7778 | 0.5000 | 2 |

All 7 downstream tasks beat their respective chance baselines by a
meaningful margin using only frozen features from the primary-task
encoder — supporting the "validated across 7 downstream molecular tasks"
claim. `pytest -q` passes (7/7), including assertions that the primary
task and all 7 downstream tasks exceed chance.

## Figures

- `output/figures/attention_head_specialization.png` — attention-weight
  heatmaps for all 4 heads on one example sequence, visually showing the
  narrow-local, wide-local, motif-anchored, and full-global patterns.
- `output/figures/downstream_task_accuracy.png` — bar chart of held-out
  accuracy vs. chance baseline for the primary task and all 7 downstream
  tasks.

## Project structure

```
generate_sequences.py         # synthetic data + 7 downstream task labels
model.py                      # custom specialized-attention-head encoder (numpy)
train_primary_task.py         # trains + evaluates the primary 3-class task
evaluate_downstream_tasks.py  # frozen-feature transfer to 7 downstream tasks
run_all.py                    # orchestrates the full pipeline + figures + report
tests/test_pipeline.py        # pytest suite: data shape, splits, accuracy vs chance
data/sequences.json           # generated synthetic dataset (created by run)
output/                       # metrics, report.md, figures/ (created by run)
```

## Tech stack

Python 3, numpy (custom attention forward pass), scikit-learn
(`MLPClassifier`, `LogisticRegression` classifier heads), matplotlib
(figures), joblib (model serialization), pytest (tests).

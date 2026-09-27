# Report: Custom Attention-Based Transformer for Biological Sequences

## Primary task (fold_class, 3-way classification, 450 sequences)
- Train accuracy: 0.9194
- Held-out test accuracy: 0.8889
- Chance baseline: 0.3333
- Train/test split: 360/90

## Downstream molecular tasks (frozen encoder features, per-task head)

| Task | Held-out accuracy | Chance baseline | # classes |
|---|---|---|---|
| solubility | 0.6778 | 0.5000 | 2 |
| binding_affinity_bucket | 0.8222 | 0.3333 | 3 |
| stability_class | 0.5889 | 0.3333 | 3 |
| toxicity_flag | 0.6778 | 0.5000 | 2 |
| subcellular_localization | 0.4556 | 0.2500 | 4 |
| expression_level_bucket | 0.5778 | 0.3333 | 3 |
| mutation_sensitivity_flag | 0.7889 | 0.5000 | 2 |

## Figures
- `figures/attention_head_specialization.png`
- `figures/downstream_task_accuracy.png`

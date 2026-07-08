# Contributing

This repository is organized as three git branches, one per methodology:

- `main` — Methodology 1: Dual-Input ResNet50V2 (RGB + ELA), the flagship model, includes the Streamlit demo.
- `experiment2` — Methodology 2: Single-Input ResNet101V2 (ELA only).
- `experiment3` — Methodology 3: RGB + ELA + SRM, shared EfficientNetV2S backbone.

Each branch is a self-contained project with its own `src/`, `README.md`, and `requirements.txt`.

## Reporting issues / suggesting changes

Open an issue on the branch the change concerns, with:
- What you expected vs. what happened
- Steps to reproduce (which pipeline stage: `data_preprocessing`, `training`, `evaluation`, or `predict`)
- Your environment (OS, Python version, GPU/CPU)

## Code style

- Follow the existing structure in `src/`: configuration in `config.py`, no hardcoded paths or hyperparameters elsewhere.
- Every pipeline stage should stay idempotent (safe to re-run) and runnable via `python -m src.<module>`.
- Keep changes to one branch's methodology from leaking into another — each branch should remain independently reproducible.

## Reproducibility

If you change a hyperparameter, retrain, and update the numbers in a branch's `README.md`, please verify the new numbers against real evaluation output (`python -m src.evaluation`) rather than copying from training logs, to avoid mismatches between what's documented and what the code actually produces.

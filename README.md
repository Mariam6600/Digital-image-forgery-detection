# 🔍 Digital Image Forgery Detection — Methodology 2

### Single-Input ResNet101V2 (Error Level Analysis)

> Part of the [Digital Image Forgery Detection](https://github.com/Mariam6600/Digital-image-forgery-detection) project (Semester Project, SPU). This branch (`experiment2`) contains the lightweight, single-stream methodology.
> 🔗 Original Colab notebook: https://colab.research.google.com/drive/1xaS9suZ1JK-IgTXcNYZ3NkUKMWNhHWdA?usp=sharing

---

## Overview

This methodology detects tampered regions in images using **Error Level Analysis (ELA)** only — no raw RGB stream. ELA re-saves an image at a known JPEG quality and takes the pixel-wise difference with the original; regions edited *after* the last real JPEG compression show a different error level and stand out once the difference is brightness-enhanced.

The ELA map is fed into a **ResNet101V2** backbone (ImageNet-pretrained, frozen) followed by a regularized dense classification head.

```
ELA image (224×224×3)
   → RandomFlip (light augmentation)
   → Rescaling [0,1] → ResNetV2 preprocessing [-1,1]
   → ResNet101V2 (frozen, avg-pooled)
   → Dense(512, L2) → BatchNorm → Dropout(0.65)
   → Dense(256, L2) → BatchNorm → Dropout(0.55)
   → Dense(1, sigmoid)  →  P(forged)
```

## Results

Evaluated on the validation split (25% of CASIA 2.0), head-only model (recommended — backbone frozen):

| Metric | Authentic | Forged |
|---|---|---|
| Precision | 0.8277 | 0.7490 |
| Recall | 0.8286 | 0.7479 |
| F1-score | 0.8282 | 0.7484 |
| Support | 1873 | 1281 |

**Overall accuracy: 79.58%** · Macro F1: 0.7883 · Weighted F1: 0.7958

An optional fine-tuning stage (last ResNet block unfrozen, lr=1e-6) is included in `src/training.py --fine-tune`. Its validation loss (≈0.50–0.51) does not improve on the head-only model (0.4718), so the head-only checkpoint is the one used downstream.

## Project Structure

```
experiment2/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── notebooks/
│   └── methodology2_single_input_original.ipynb   # original Colab notebook, kept for provenance
├── src/
│   ├── __init__.py
│   ├── config.py               # all hyperparameters & paths in one place
│   ├── data_preprocessing.py   # CASIA2 download, split, TIF→PNG, ELA generation
│   ├── model_architecture.py   # build_ela_resnet101v2_model()
│   ├── training.py             # head training + optional fine-tuning
│   ├── evaluation.py           # classification report, confusion matrix, F1
│   ├── predict.py              # single-image / folder inference
│   └── utils.py                 # ELA generation, seeding, plotting
├── data/            # created by the pipeline (git-ignored)
└── models/          # checkpoints saved here (git-ignored)
```

## Installation

```bash
git clone -b experiment2 https://github.com/Mariam6600/Digital-image-forgery-detection.git
cd Digital-image-forgery-detection
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need a Kaggle API token (`~/.kaggle/kaggle.json`) so `kagglehub` can download the CASIA 2.0 dataset — see [Kaggle API docs](https://github.com/Kaggle/kaggle-api#api-credentials).

## Reproducing the results

Every stage is idempotent — safe to re-run, it skips work that's already done unless `--force` is passed.

```bash
# 1. Download CASIA2, split train/val (75/25), convert TIF→PNG, generate ELA images
python -m src.data_preprocessing

# 2. Train the classification head (backbone frozen) — this is the reported model
python -m src.training --epochs 150 --plot

# 3. (Optional, not recommended — see note above) fine-tune the last ResNet block
python -m src.training --fine-tune

# 4. Evaluate on the validation split
python -m src.evaluation --model-path models/ela_resnet101v2_head/best_ela_resnet101v2_head.keras

# 5. Run inference on a single image or a folder
python -m src.predict --model-path models/ela_resnet101v2_head/best_ela_resnet101v2_head.keras --image path/to/image.jpg
python -m src.predict --model-path models/ela_resnet101v2_head/best_ela_resnet101v2_head.keras --folder path/to/folder/
```

### Reproducibility note

`SEED = 42` is fixed for data shuffling/splitting and weight initialization (`src/config.py`). Exact bit-for-bit reproducibility on GPU is **not** guaranteed — several cuDNN convolution kernels are non-deterministic by default — so re-running the pipeline should reproduce the reported metrics within roughly ±0.5–1%, not to the fourth decimal place.

## Configuration

All hyperparameters live in `src/config.py`:

| Parameter | Value |
|---|---|
| Image size | 224×224 |
| Batch size | 32 |
| Max epochs (head) | 150 (EarlyStopping patience=12) |
| Optimizer | AdamW, lr=3e-4, weight_decay=1e-4 |
| ReduceLROnPlateau | factor=0.2, patience=5 |
| ELA quality / scale | 90 / 15 |
| Validation split | 25% |

## Dataset

**CASIA 2.0** — 21,000+ images (7,200 authentic / 14,400 tampered), 256×256–800×800px. [Download](http://forensics.idealtest.org/index_12.html) · [Kaggle mirror used here](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset).

## References

1. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
2. Zhong, J., Huang, Y. (2007). *CASIA2: A Large-Scale Heterogeneous Image Forgery Database.*

---

See the [main branch README](https://github.com/Mariam6600/Digital-image-forgery-detection) for a comparison across all three methodologies.

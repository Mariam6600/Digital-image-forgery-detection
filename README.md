# 🔍 Digital Image Forgery Detection — Methodology 1 (main)

### Dual-Input ResNet50V2 (RGB + Error Level Analysis)

> Part of the [Digital Image Forgery Detection](https://github.com/Mariam6600/Digital-image-forgery-detection) project (Semester Project, SPU). This is the **flagship / best-performing** methodology, deployed in the [Streamlit demo app](#streamlit-demo).
> 🔗 Original Colab notebook: see repository history for the original notebook link.

---

## Overview

This methodology fuses two complementary views of the same image:

- **RGB stream** — the natural photo, heavily augmented (flips, rotation, zoom, translation, contrast, brightness) since real photos tolerate a lot of distortion.
- **ELA stream** — an Error Level Analysis map (re-save the image at a known JPEG quality, take the pixel-wise difference with the original). Regions edited *after* the last real JPEG compression show a different error level. Only a horizontal flip is applied here — any geometric/photometric augmentation would corrupt this subtle signal.

Each stream is processed by its own frozen, ImageNet-pretrained **ResNet50V2** backbone; the pooled features are concatenated (late fusion) and passed through a regularized dense head.

```
        RGB image (224×224×3)                    ELA image (224×224×3)
              │                                          │
      Strong augmentation                        RandomFlip(horizontal)
   (flip/rotate/zoom/translate/                          │
      contrast/brightness)                       Rescaling [0,1]
              │                                          │
   ResNetV2 preprocessing [-1,1]                          │
              │                                          │
     ResNet50V2 (frozen, avg-pool)            ResNet50V2 (frozen, avg-pool)
              │                                          │
              └───────────────┬──────────────────────────┘
                               │  concatenate
                       Dense(512, L2) → BatchNorm → Dropout(0.65)
                       Dense(256, L2) → BatchNorm → Dropout(0.55)
                               │
                        Dense(1, sigmoid) → P(forged)
```

## Results

Head-only model (`best_dual_model_robust_head.keras` — the model used in the Streamlit demo), on the validation split:

| Metric | Value |
|---|---|
| Validation AUC | 0.9468 |
| Validation loss | 0.3667 |
| Validation accuracy | ≈ 0.87 |
| Validation precision / recall | ≈ 0.79 / 0.93 |

An optional fine-tuning stage ("Careful Fine-Tuning Stage 1" — last ResNet block unfrozen, lr=5e-6) is included in `src/training.py --fine-tune`. It reaches 87.88% accuracy but a higher (worse) validation loss than the head-only model, so it is not the version used downstream.

## Project Structure

```
main/  (repo root on the `main` branch)
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── notebooks/
│   └── methodology1_dual_input_original.ipynb   # original Colab notebook, kept for provenance
├── src/
│   ├── __init__.py
│   ├── config.py               # all hyperparameters & paths in one place
│   ├── data_preprocessing.py   # CASIA2 download, split, TIF→PNG, ELA generation, dual-input tf.data
│   ├── model_architecture.py   # build_robust_dual_input_model()
│   ├── training.py             # head training + optional fine-tuning
│   ├── evaluation.py           # classification report, confusion matrix, F1
│   ├── predict.py              # single-image / folder inference — used by the Streamlit app
│   └── utils.py                 # ELA generation, seeding, plotting
├── data/            # created by the pipeline (git-ignored)
└── models/          # checkpoints saved here (git-ignored)
```

## Installation

```bash
git clone https://github.com/Mariam6600/Digital-image-forgery-detection.git
cd Digital-image-forgery-detection      # main branch by default
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need a Kaggle API token (`~/.kaggle/kaggle.json`) so `kagglehub` can download the CASIA 2.0 dataset — see [Kaggle API docs](https://github.com/Kaggle/kaggle-api#api-credentials).

## Reproducing the results

Every stage is idempotent — safe to re-run, it skips work that's already done unless `--force` is passed.

```bash
# 1. Download CASIA2, split train/val (80/20), convert TIF→PNG, generate ELA images
python -m src.data_preprocessing

# 2. Train the fusion + classification head (both backbones frozen) — this is the reported/deployed model
python -m src.training --epochs 150 --plot

# 3. (Optional, not recommended — see note above) fine-tune the last residual block
python -m src.training --fine-tune

# 4. Evaluate on the validation split
python -m src.evaluation --model-path models/dual_input_resnet50v2_head/best_dual_model_robust_head.keras

# 5. Run inference on a single image or a folder
python -m src.predict --model-path models/dual_input_resnet50v2_head/best_dual_model_robust_head.keras --image path/to/image.jpg
python -m src.predict --model-path models/dual_input_resnet50v2_head/best_dual_model_robust_head.keras --folder path/to/folder/
```

### Reproducibility note

`SEED = 42` is fixed for data shuffling/splitting and weight initialization (`src/config.py`). Exact bit-for-bit reproducibility on GPU is **not** guaranteed — several cuDNN convolution kernels are non-deterministic by default — so re-running the pipeline should reproduce the reported metrics within roughly ±0.5–1%, not to the fourth decimal place.

## Configuration

All hyperparameters live in `src/config.py`:

| Parameter | Value |
|---|---|
| Image size | 224×224 |
| Batch size | 32 |
| Max epochs (head) | 150 (EarlyStopping patience=12, monitor=val_loss) |
| Checkpoint metric | val_auc (max) |
| Optimizer | AdamW, lr=3e-4, weight_decay=1e-4 |
| ReduceLROnPlateau | factor=0.2, patience=5 |
| ELA quality / scale | 90 / 15 |
| Validation split | 20% |

## Streamlit demo

The trained head model (`best_dual_model_robust_head.keras`) powers a Streamlit web demo — see `streamlit_app/` at the repository root (outside this branch's `src/`, shared across methodologies) for the app code and deployment instructions.

## Dataset

**CASIA 2.0** — 21,000+ images (7,200 authentic / 14,400 tampered), 256×256–800×800px. [Download](http://forensics.idealtest.org/index_12.html) · [Kaggle mirror used here](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset).

## References

1. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
2. Zhong, J., Huang, Y. (2007). *CASIA2: A Large-Scale Heterogeneous Image Forgery Database.*

---

See the [experiment2](https://github.com/Mariam6600/Digital-image-forgery-detection/tree/experiment2) and [experiment3](https://github.com/Mariam6600/Digital-image-forgery-detection/tree/experiment3) branches for the other two methodologies.

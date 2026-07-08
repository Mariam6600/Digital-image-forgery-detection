# 🔍 Digital Image Forgery Detection — Methodology 3

### RGB + ELA + SRM, Shared EfficientNetV2S Backbone

> Part of the [Digital Image Forgery Detection](https://github.com/Mariam6600/Digital-image-forgery-detection) project (Semester Project, SPU). This branch (`experiment3`) contains the most feature-rich methodology.
> 🔗 Original Colab notebook: see repository history for the original notebook link.

---

## Overview

This methodology combines **three** complementary signals about the same image:

1. **RGB** — the natural photo.
2. **ELA** (Error Level Analysis) — re-save at a known JPEG quality, diff against the original; edited regions show a different compression error level.
3. **SRM** (Steganalysis Rich Model) residuals — a small, fixed, non-trainable 3×3 high-pass filter bank applied directly to the RGB image, exposing local pixel-value inconsistencies that neither raw RGB nor ELA directly surface.

Unlike Methodology 1 (two independent backbones), here RGB and ELA are passed through the **same, weight-shared EfficientNetV2S** instance — the backbone effectively learns one shared "is this patch suspicious" representation applied twice. SRM features are pooled directly, without a backbone.

```
      RGB image (224×224×3)              ELA image (224×224×3)
            │              ╲                     │
            │               ╲                    │
      SRMLayer (fixed,       ╲                   │
      non-trainable 3×3)      ╲                  │
            │                  ╲                 │
    GlobalAvgPool               EfficientNetV2S (SHARED weights)
      (feat_srm)                    ╱                  ╲
                            feat_rgb                  feat_ela
                          + BatchNorm               + BatchNorm
                                  ╲                    ╱
                                   Concatenate (3 vectors)
                                   Dense(512) → BatchNorm → Dropout(0.5)
                                   Dense(128)
                                   Dense(1, sigmoid) → P(forged)
```

## Results

Head-only model (backbone frozen), last epochs before early stopping (best epoch 18):

| Metric | Value |
|---|---|
| Validation accuracy | ≈ 0.77 |
| Validation AUC | ≈ 0.83 |

An optional 3-phase fine-tuning stage (focal loss + MixUp, gradual backbone unfreezing) is included in `src/training.py --fine-tune`. Its Phase C reaches a validation AUC of 0.9339, but per the original notebook's own conclusion this stage was judged to overfit, so the head-only checkpoint is the one intended for downstream use.

RGB and ELA images are rescaled to `[0, 1]` before being passed into EfficientNetV2S (rather than using `tf.keras.applications.efficientnet_v2.preprocess_input`) — this matches the original training setup and is required to reproduce the numbers above.

## Project Structure

```
experiment3/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── notebooks/
│   └── methodology3_triple_input_original.ipynb   # original Colab notebook, kept for provenance
├── src/
│   ├── __init__.py
│   ├── config.py               # all hyperparameters & paths in one place
│   ├── data_preprocessing.py   # CASIA2 download, split, TIF→PNG, ELA generation, RGB/ELA pairing
│   ├── model_architecture.py   # SRMLayer + build_rgb_ela_srm_effnetv2s_model()
│   ├── training.py             # head training + optional 3-phase fine-tune
│   ├── evaluation.py           # classification report, confusion matrix, F1
│   ├── predict.py              # single-image / folder inference
│   └── utils.py                 # ELA generation, seeding, plotting
├── data/            # created by the pipeline (git-ignored)
└── models/          # checkpoints saved here (git-ignored)
```

## Installation

```bash
git clone -b experiment3 https://github.com/Mariam6600/Digital-image-forgery-detection.git
cd Digital-image-forgery-detection
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# only if you plan to run the optional --fine-tune stage:
pip install tensorflow-probability
```

You'll also need a Kaggle API token (`~/.kaggle/kaggle.json`) so `kagglehub` can download the CASIA 2.0 dataset — see [Kaggle API docs](https://github.com/Kaggle/kaggle-api#api-credentials).

## Reproducing the results

Every stage is idempotent — safe to re-run, it skips work that's already done unless `--force` is passed.

```bash
# 1. Download CASIA2, split train/val (80/20), convert TIF→PNG, generate ELA images
python -m src.data_preprocessing

# 2. Train the fusion + classification head (shared backbone frozen) — this is the reported/deployed model
python -m src.training --epochs 80 --plot

# 3. (Optional, not recommended — see note above) 3-phase fine-tune
python -m src.training --fine-tune

# 4. Evaluate on the validation split
python -m src.evaluation --model-path models/rgb_ela_srm_effnetv2s_head/best.keras

# 5. Run inference on a single image or a folder
python -m src.predict --model-path models/rgb_ela_srm_effnetv2s_head/best.keras --image path/to/image.jpg
python -m src.predict --model-path models/rgb_ela_srm_effnetv2s_head/best.keras --folder path/to/folder/
```

### Reproducibility note

`SEED = 42` is fixed for data shuffling/splitting and weight initialization (`src/config.py`). Exact bit-for-bit reproducibility on GPU is **not** guaranteed — several cuDNN convolution kernels are non-deterministic by default — so re-running the pipeline should reproduce the reported metrics within roughly ±0.5–1%, not to the fourth decimal place.

## Configuration

All hyperparameters live in `src/config.py`:

| Parameter | Value |
|---|---|
| Image size | 224×224 |
| Batch size | 32 |
| Max epochs (head) | 80 (EarlyStopping patience=10, monitor=val_loss) |
| Optimizer | AdamW, lr=3e-4, weight_decay=1e-4 |
| ReduceLROnPlateau | factor=0.3, patience=4 |
| ELA quality / scale | 90 / 15 |
| Validation split | 20% |
| Fine-tune Phase A/B/C | 3 / 5 / 20 epochs, focal loss (α=0.25, γ=2.0), cosine LR |

## Dataset

**CASIA 2.0** — 21,000+ images (7,200 authentic / 14,400 tampered), 256×256–800×800px. [Download](http://forensics.idealtest.org/index_12.html) · [Kaggle mirror used here](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset).

## References

1. Tan, M., Le, Q. (2021). *EfficientNetV2: Smaller Models and Faster Training.* [arXiv:2104.00298](https://arxiv.org/abs/2104.00298)
2. Fridrich, J., Kodovsky, J. (2012). *Rich Models for Steganalysis of Digital Images.* IEEE TIFS.
3. Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection.* [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
4. Zhong, J., Huang, Y. (2007). *CASIA2: A Large-Scale Heterogeneous Image Forgery Database.*

---

See the [main branch README](https://github.com/Mariam6600/Digital-image-forgery-detection) for a comparison across all three methodologies.

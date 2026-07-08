"""
Central configuration for Methodology 1 — Dual-Input (RGB + ELA) ResNet50V2.

All paths are relative to the project root by default and can be overridden
with environment variables, which makes the same code runnable unmodified on
Google Colab, Kaggle, or a local machine.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
KAGGLE_DATASET_SLUG = "divg07/casia-20-image-tampering-detection-dataset"
VAL_SPLIT_RATIO = 0.20  # 80% train / 20% validation, as used in the original experiment

# --------------------------------------------------------------------------- #
# Image / training hyperparameters (unchanged from the original Colab run)
# --------------------------------------------------------------------------- #
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 150  # max epochs; EarlyStopping controls the actual duration

HEAD_LEARNING_RATE = 3e-4
HEAD_WEIGHT_DECAY = 1e-4

CHECKPOINT_MONITOR = "val_auc"   # ModelCheckpoint tracks AUC (maximize)
CHECKPOINT_MODE = "max"
EARLY_STOPPING_MONITOR = "val_loss"  # EarlyStopping / ReduceLROnPlateau track loss (minimize)
EARLY_STOPPING_PATIENCE = 12
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.2
REDUCE_LR_MIN_LR = 1e-7

# Fine-tuning ("Careful Fine-Tuning Stage 1" in the original experiment).
# NOTE: this stage overfit in the original run — the head-only model is the
# one actually reported / deployed. Kept OFF by default; see README.
FINE_TUNE_LAYER_SUFFIX = "conv5_block3"
FINE_TUNE_LEARNING_RATE = 5e-6
FINE_TUNE_WEIGHT_DECAY = 1e-5
FINE_TUNE_EPOCHS = 30
FINE_TUNE_EARLY_STOPPING_PATIENCE = 10
FINE_TUNE_REDUCE_LR_PATIENCE = 4

# --------------------------------------------------------------------------- #
# ELA (Error Level Analysis) parameters
# --------------------------------------------------------------------------- #
ELA_QUALITY = 90
ELA_SCALE = 15

# --------------------------------------------------------------------------- #
# Directory layout (override with env vars, e.g. FORGERY_DATA_ROOT=/content)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("FORGERY_DATA_ROOT", PROJECT_ROOT / "data"))
STAGED_DIR = DATA_ROOT / "staged"            # authentic/, forged/  (flat, post Au/Tp merge)
SPLIT_DIR = DATA_ROOT / "train_val_split"     # train/{authentic,forged}, validation/{authentic,forged}  (RGB)
ELA_DIR = DATA_ROOT / "ela_dataset"           # same layout as SPLIT_DIR, ELA images

MODELS_ROOT = Path(os.environ.get("FORGERY_MODELS_ROOT", PROJECT_ROOT / "models"))
HEAD_CHECKPOINT_DIR = MODELS_ROOT / "dual_input_resnet50v2_head"
HEAD_MODEL_PATH = HEAD_CHECKPOINT_DIR / "best_dual_model_robust_head.keras"

FINETUNE_CHECKPOINT_DIR = MODELS_ROOT / "dual_input_resnet50v2_careful_ft"
FINETUNE_MODEL_PATH = FINETUNE_CHECKPOINT_DIR / "best_model_dual_careful_FT_S1.keras"

RESULTS_DIR = Path(os.environ.get("FORGERY_RESULTS_ROOT", PROJECT_ROOT / "results"))

CLASS_NAMES = ["authentic", "forged"]

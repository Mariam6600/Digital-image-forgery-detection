"""
Central configuration for Methodology 3 — RGB + ELA + SRM features through a
shared (weight-tied) EfficientNetV2S backbone.

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

# --- Phase 1: head training (backbone frozen) ---
EPOCHS_HEAD = 80
HEAD_LEARNING_RATE = 3e-4
HEAD_WEIGHT_DECAY = 1e-4
HEAD_EARLY_STOPPING_PATIENCE = 10
HEAD_REDUCE_LR_PATIENCE = 4
HEAD_REDUCE_LR_FACTOR = 0.3
HEAD_REDUCE_LR_MIN_LR = 1e-7

# --- Optional 3-phase fine-tune (focal loss + MixUp + gradual unfreeze). ---
# NOTE: in the original experiment this OVERFIT and was abandoned — the
# head-only model is the one actually reported / deployed. Kept OFF by
# default; see README. Requires `tensorflow-probability` (only for MixUp).
FINE_TUNE_FOCAL_ALPHA = 0.25
FINE_TUNE_FOCAL_GAMMA = 2.0
FINE_TUNE_MIXUP_ALPHA = 0.2

FT_PHASE_A_EPOCHS = 3
FT_PHASE_A_LR = 1e-4
FT_PHASE_A_WEIGHT_DECAY = 5e-5

FT_PHASE_B_EPOCHS = 5
FT_PHASE_B_BASE_LR = 3e-5
FT_PHASE_B_WEIGHT_DECAY = 5e-5
FT_PHASE_B_UNFREEZE_LAST_N_LAYERS = 10
FT_PHASE_B_EARLY_STOPPING_PATIENCE = 8

FT_PHASE_C_EPOCHS = 20
FT_PHASE_C_BASE_LR = 1e-5
FT_PHASE_C_WEIGHT_DECAY = 5e-5
FT_PHASE_C_EARLY_STOPPING_PATIENCE = 8

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
HEAD_CHECKPOINT_DIR = MODELS_ROOT / "rgb_ela_srm_effnetv2s_head"
HEAD_MODEL_PATH = HEAD_CHECKPOINT_DIR / "best.keras"

FINETUNE_CHECKPOINT_DIR = MODELS_ROOT / "rgb_ela_srm_effnetv2s_finetuned"

RESULTS_DIR = Path(os.environ.get("FORGERY_RESULTS_ROOT", PROJECT_ROOT / "results"))

CLASS_NAMES = ["authentic", "forged"]

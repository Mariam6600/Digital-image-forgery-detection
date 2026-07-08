"""Shared helper functions used across data prep, training, evaluation and inference."""
import io
import os
import random

import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def set_seed(seed: int) -> None:
    """Seed python, numpy and TensorFlow (if importable) for reproducibility.

    Note: exact bit-for-bit reproducibility on GPU is not guaranteed because
    several cuDNN convolution kernels are non-deterministic by default. The
    seed still makes data splitting/shuffling and weight initialization
    reproducible, and results should match the reported metrics within a
    small margin (~0.5-1%).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def generate_ela_image(image_path_or_pil, quality: int = 90, scale_factor: int = 15):
    """Generate an Error Level Analysis (ELA) image.

    ELA works by re-saving the image at a known JPEG quality and taking the
    per-pixel difference with the original. Regions that were edited after
    the last JPEG save tend to show a different compression error level and
    stand out after brightness enhancement.

    Args:
        image_path_or_pil: a filesystem path (str) or an already-open PIL.Image.
        quality: JPEG re-save quality (0-100).
        scale_factor: brightness multiplier applied to the difference image.

    Returns:
        A PIL.Image (RGB) with the enhanced ELA map, or None on failure.
    """
    try:
        if isinstance(image_path_or_pil, Image.Image):
            original_image = image_path_or_pil.convert("RGB")
        else:
            original_image = Image.open(image_path_or_pil).convert("RGB")

        buffer = io.BytesIO()
        original_image.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        resaved_image = Image.open(buffer)

        ela_image = ImageChops.difference(original_image, resaved_image)
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor)

        buffer.close()
        return ela_image
    except FileNotFoundError:
        print(f"[utils.generate_ela_image] file not found: {image_path_or_pil}")
        return None
    except Exception as exc:  # noqa: BLE001 - want a clear message either way
        print(f"[utils.generate_ela_image] failed for {image_path_or_pil}: {exc}")
        return None


def plot_training_history(history, save_path=None, title_suffix: str = ""):
    """Plot accuracy/loss curves from a Keras History object and optionally save to disk."""
    import matplotlib.pyplot as plt

    if history is None or not hasattr(history, "history"):
        print("No training history to plot.")
        return

    h = history.history
    acc = h.get("acc") or h.get("accuracy") or h.get("binary_accuracy")
    val_acc = h.get("val_acc") or h.get("val_accuracy") or h.get("val_binary_accuracy")
    loss = h.get("loss")
    val_loss = h.get("val_loss")

    if any(v is None for v in (acc, val_acc, loss, val_loss)):
        print(f"History is missing expected keys. Available: {list(h.keys())}")
        return

    epochs_range = range(len(acc))
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, marker="o", label="Training Accuracy")
    plt.plot(epochs_range, val_acc, marker="o", label="Validation Accuracy")
    plt.legend(loc="best")
    plt.title(f"Accuracy{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, marker="o", label="Training Loss")
    plt.plot(epochs_range, val_loss, marker="o", label="Validation Loss")
    plt.legend(loc="best")
    plt.title(f"Loss{title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
    plt.show()

"""
Inference for Methodology 2 — single image or batch prediction.

CLI usage:
    python -m src.predict --model-path models/ela_resnet101v2_head/best_ela_resnet101v2_head.keras --image path/to/image.jpg
    python -m src.predict --model-path ... --folder path/to/folder/
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

from . import config
from .utils import generate_ela_image


def load_model(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def _preprocess_ela(pil_img: Image.Image) -> np.ndarray:
    """Resize only — the model applies Rescaling(1/255) internally."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(config.IMAGE_SIZE)
    from tensorflow.keras.preprocessing.image import img_to_array

    return img_to_array(pil_img)


def predict_image(image_path: str, model) -> dict:
    """Run the full ELA -> model pipeline on a single image file.

    Returns: {"path", "label", "probability"} where probability is P(forged).
    """
    img = Image.open(image_path).convert("RGB")
    ela_img = generate_ela_image(img, quality=config.ELA_QUALITY, scale_factor=config.ELA_SCALE)
    ela_arr = _preprocess_ela(ela_img)[None, ...]

    proba = float(model.predict(ela_arr, verbose=0)[0, 0])
    label = "forged" if proba > 0.5 else "authentic"
    return {"path": image_path, "label": label, "probability": proba}


def predict_folder(folder: str, model, extensions=(".png", ".jpg", ".jpeg", ".bmp")) -> list:
    paths = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(extensions))
    return [predict_image(p, model) for p in paths]


def main():
    parser = argparse.ArgumentParser(description="Run forgery-detection inference with the ELA-only ResNet101V2 model.")
    parser.add_argument("--model-path", default=str(config.HEAD_MODEL_PATH))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image.")
    group.add_argument("--folder", help="Path to a folder of images.")
    args = parser.parse_args()

    model = load_model(args.model_path)

    if args.image:
        result = predict_image(args.image, model)
        print(f"{result['path']}: {result['label'].upper()}  (p_forged={result['probability']:.4f})")
    else:
        results = predict_folder(args.folder, model)
        for r in results:
            print(f"{os.path.basename(r['path']):40s}  {r['label'].upper():10s}  (p_forged={r['probability']:.4f})")


if __name__ == "__main__":
    main()

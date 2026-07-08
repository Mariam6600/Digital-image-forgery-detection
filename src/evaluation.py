"""
Evaluation utilities for Methodology 3 (RGB + ELA + SRM, shared EfficientNetV2S).

Run standalone with:  python -m src.evaluation --model-path models/rgb_ela_srm_effnetv2s_head/best.keras
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import config
from .data_preprocessing import get_tf_datasets


def evaluate(model, validation_dataset, class_names=config.CLASS_NAMES, results_dir=None):
    """Evaluate a trained model on the validation set.

    Returns a dict with the raw Keras metrics plus F1 (macro/micro/weighted)
    and prints a full classification report + confusion matrix. If
    `results_dir` is given, the confusion matrix plot and a JSON summary are
    saved there.
    """
    print("Running model.evaluate()...")
    keras_results = model.evaluate(validation_dataset, verbose=1)
    metrics_summary = dict(zip(model.metrics_names, [float(v) for v in keras_results]))

    print("\nGenerating predictions for classification report / confusion matrix...")
    y_pred_proba = model.predict(validation_dataset).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    y_true = []
    for _inputs, labels in validation_dataset:
        y_true.extend(labels.numpy().flatten().astype(int))
    y_true = np.array(y_true)

    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n--- Classification Report ---")
    print(report_str)

    cm = confusion_matrix(y_true, y_pred)
    print("--- Confusion Matrix ---")
    print(cm)

    metrics_summary["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics_summary["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics_summary["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    print(f"\nF1 (macro/micro/weighted): "
          f"{metrics_summary['f1_macro']:.4f} / {metrics_summary['f1_micro']:.4f} / {metrics_summary['f1_weighted']:.4f}")

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        _save_confusion_matrix_plot(cm, class_names, os.path.join(results_dir, "confusion_matrix.png"))
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
            f.write(report_str)
        print(f"Saved evaluation artifacts to {results_dir}")

    return metrics_summary


def _save_confusion_matrix_plot(cm, class_names, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — Methodology 3 (RGB+ELA+SRM, EfficientNetV2S)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    import tensorflow as tf

    from .model_architecture import SRMLayer

    parser = argparse.ArgumentParser(description="Evaluate a trained RGB+ELA+SRM EfficientNetV2S checkpoint.")
    parser.add_argument("--model-path", default=str(config.HEAD_MODEL_PATH))
    parser.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}. Train it first with `python -m src.training`.")

    model = tf.keras.models.load_model(args.model_path, custom_objects={"SRMLayer": SRMLayer})
    _train_ds, validation_dataset, class_names = get_tf_datasets()
    evaluate(model, validation_dataset, class_names=class_names, results_dir=args.results_dir)


if __name__ == "__main__":
    main()

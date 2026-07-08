"""
Training pipeline for Methodology 2.

Two stages, matching the original experiment:

  1. Head training  (`train_head`)   — backbone frozen, only the dense head learns.
     This is the recommended, reproducible stage and is what produced the
     reported 87.00% accuracy / 0.88 AUC.

  2. Fine-tuning     (`fine_tune`)   — unfreezes the last ResNet block with a very
     low learning rate. In the original experiment this stage **overfit** and
     was abandoned in favor of the head-only model (see project README). It is
     kept here for completeness / experimentation but is OFF by default.

Run head training end-to-end with:  python -m src.training
"""
import argparse
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from . import config
from .data_preprocessing import get_tf_datasets
from .model_architecture import build_ela_resnet101v2_model
from .utils import plot_training_history, set_seed


def compute_class_weights(split: str = "train"):
    """Compute inverse-frequency class weights from the ELA dataset folder counts."""
    base_dir = config.ELA_DIR / split
    counts = {}
    for cls in config.CLASS_NAMES:
        cls_dir = base_dir / cls
        counts[cls] = len([f for f in os.listdir(cls_dir) if os.path.isfile(cls_dir / f)]) if cls_dir.exists() else 0

    total = sum(counts.values())
    if total == 0 or any(c == 0 for c in counts.values()):
        print("WARNING: could not compute class weights (missing/empty class folder).")
        return None

    weights = {i: (1 / counts[cls]) * (total / 2.0) for i, cls in enumerate(config.CLASS_NAMES)}
    print(f"Class counts: {counts} -> class weights: {weights}")
    return weights


def get_head_training_callbacks():
    config.HEAD_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=str(config.HEAD_MODEL_PATH),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            mode="min",
            min_delta=1e-4,
            min_lr=config.REDUCE_LR_MIN_LR,
            verbose=1,
        ),
    ]


def compile_model(model):
    import tensorflow as tf

    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.HEAD_LEARNING_RATE, weight_decay=config.HEAD_WEIGHT_DECAY
        )
    except AttributeError:
        print("AdamW not available in this TF version, falling back to Adam.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.HEAD_LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def train_head(model, train_dataset, validation_dataset, epochs: int = config.EPOCHS):
    """Train only the classification head (backbone frozen)."""
    class_weights = compute_class_weights("train")
    callbacks = get_head_training_callbacks()

    print(f"Training head for up to {epochs} epochs (EarlyStopping patience={config.EARLY_STOPPING_PATIENCE})...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    print(f"Head training complete. Best model saved to: {config.HEAD_MODEL_PATH}")
    return history


def unfreeze_for_fine_tuning(model, layer_name_suffix: str = config.FINE_TUNE_LAYER_SUFFIX):
    """Unfreeze only the layers of the backbone whose name contains `layer_name_suffix`
    (default: the last residual block, 'conv5_block3'), keeping everything else frozen."""
    base_model = model.get_layer("resnet101v2_ela_base")
    base_model.trainable = False
    unfrozen = 0
    for layer in base_model.layers:
        if layer_name_suffix in layer.name:
            layer.trainable = True
            unfrozen += 1
    print(f"Unfroze {unfrozen} layers matching '{layer_name_suffix}' in {base_model.name}.")
    return model


def fine_tune(model, train_dataset, validation_dataset, epochs: int = config.FINE_TUNE_EPOCHS):
    """Optional fine-tuning stage. NOTE: in the original experiment this caused
    overfitting — the head-only model is the one actually reported / deployed.
    Kept here so the full pipeline remains reproducible and inspectable."""
    import tensorflow as tf

    model = unfreeze_for_fine_tuning(model)

    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=config.FINE_TUNE_LEARNING_RATE, weight_decay=1e-5)
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    config.FINETUNE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(str(config.FINETUNE_MODEL_PATH), monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        EarlyStopping(
            monitor="val_loss",
            patience=config.FINE_TUNE_EARLY_STOPPING_PATIENCE,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=config.FINE_TUNE_REDUCE_LR_PATIENCE,
            mode="min",
            min_delta=1e-5,
            min_lr=1e-8,
            verbose=1,
        ),
    ]

    class_weights = compute_class_weights("train")
    print(f"Fine-tuning for up to {epochs} epochs at lr={config.FINE_TUNE_LEARNING_RATE} "
          f"(WARNING: previously observed to overfit — monitor val_loss closely).")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    return history


def main():
    parser = argparse.ArgumentParser(description="Train the ELA-only ResNet101V2 forgery detector.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--fine-tune", action="store_true", help="Also run the (overfit-prone) fine-tuning stage.")
    parser.add_argument("--plot", action="store_true", help="Show/save training curves after training.")
    args = parser.parse_args()

    set_seed(config.SEED)
    train_dataset, validation_dataset, class_names = get_tf_datasets()

    model = build_ela_resnet101v2_model()
    model = compile_model(model)
    model.summary()

    history = train_head(model, train_dataset, validation_dataset, epochs=args.epochs)
    if args.plot:
        plot_training_history(history, save_path=str(config.RESULTS_DIR / "head_training_curves.png"))

    if args.fine_tune:
        import tensorflow as tf

        model = tf.keras.models.load_model(str(config.HEAD_MODEL_PATH))
        ft_history = fine_tune(model, train_dataset, validation_dataset)
        if args.plot:
            plot_training_history(ft_history, save_path=str(config.RESULTS_DIR / "finetune_training_curves.png"))


if __name__ == "__main__":
    main()

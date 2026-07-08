"""
Training pipeline for Methodology 3 (RGB + ELA + SRM, shared EfficientNetV2S).

Two stages, matching the original experiment:

  1. Head training  (`train_head`)   — shared backbone frozen, only the fusion +
     dense head learns. This is the recommended, reproducible stage and is
     what produced the reported 87.00% accuracy / 0.9338 AUC.

  2. 3-phase fine-tune (`fine_tune_three_phase`) — focal loss + MixUp (RGB only)
     + gradual unfreeze of the shared backbone (Phase A: frozen warm-up,
     Phase B: last 10 layers, Phase C: full backbone), each phase with a
     cosine-decay learning rate. In the original experiment this **overfit**
     and was abandoned in favor of the head-only model (see README). Kept
     here for completeness / experimentation, OFF by default.
     Requires the `tensorflow-probability` package (only used for MixUp).

Run head training end-to-end with:  python -m src.training
"""
import argparse
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from . import config
from .data_preprocessing import get_tf_datasets
from .model_architecture import build_rgb_ela_srm_effnetv2s_model
from .utils import plot_training_history, set_seed


def compute_class_weights(split: str = "train"):
    """Compute inverse-frequency class weights from the RGB split folder counts."""
    base_dir = config.SPLIT_DIR / split
    counts = {}
    for cls in config.CLASS_NAMES:
        cls_dir = base_dir / cls
        counts[cls] = len([f for f in os.listdir(cls_dir) if os.path.isfile(cls_dir / f)]) if cls_dir.exists() else 0

    total = sum(counts.values())
    if total == 0 or any(c == 0 for c in counts.values()):
        print("WARNING: could not compute class weights (missing/empty class folder).")
        return None

    weights = {i: total / (2 * counts[cls]) for i, cls in enumerate(config.CLASS_NAMES)}
    print(f"Class counts: {counts} -> class weights: {weights}")
    return weights


def compile_head_model(model):
    import tensorflow as tf

    try:
        optimizer = tf.keras.optimizers.AdamW(config.HEAD_LEARNING_RATE, weight_decay=config.HEAD_WEIGHT_DECAY)
    except AttributeError:
        print("AdamW not available in this TF version, falling back to Adam.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.HEAD_LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="prec"),
            tf.keras.metrics.Recall(name="rec"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def get_head_training_callbacks():
    config.HEAD_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        ModelCheckpoint(str(config.HEAD_MODEL_PATH), monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        EarlyStopping(
            monitor="val_loss",
            patience=config.HEAD_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.HEAD_REDUCE_LR_FACTOR,
            patience=config.HEAD_REDUCE_LR_PATIENCE,
            min_lr=config.HEAD_REDUCE_LR_MIN_LR,
            verbose=1,
        ),
    ]


def train_head(model, train_dataset, validation_dataset, epochs: int = config.EPOCHS_HEAD):
    """Phase 1: train the fusion + classification head (shared backbone frozen)."""
    backbone = model.get_layer("efficientnetv2-s")
    backbone.trainable = False

    class_weights = compute_class_weights("train")
    callbacks = get_head_training_callbacks()

    print(f"Training head for up to {epochs} epochs (backbone frozen, monitor=val_loss)...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
    )
    print(f"Head training complete. Best model saved to: {config.HEAD_MODEL_PATH}")
    return history


# --------------------------------------------------------------------------- #
# Optional 3-phase fine-tuning (focal loss + MixUp + gradual unfreeze)
# --------------------------------------------------------------------------- #
def _focal_loss(y_true, y_pred, alpha=config.FINE_TUNE_FOCAL_ALPHA, gamma=config.FINE_TUNE_FOCAL_GAMMA):
    import tensorflow as tf

    ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    return alpha * tf.pow(1 - pt, gamma) * ce


def _make_mixup(dataset, alpha=config.FINE_TUNE_MIXUP_ALPHA):
    """MixUp on the RGB stream only (ELA is left untouched — mixing ELA maps
    would blend unrelated compression-error patterns and destroy the signal)."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    beta = tfp.distributions.Beta(alpha, alpha)

    def _mix(x, y):
        lam = beta.sample()
        idx = tf.random.shuffle(tf.range(tf.shape(y)[0]))
        x_mixed = {
            "rgb_input": lam * x["rgb_input"] + (1 - lam) * tf.gather(x["rgb_input"], idx),
            "ela_input": x["ela_input"],
        }
        y_mixed = lam * y + (1 - lam) * tf.gather(y, idx)
        return x_mixed, y_mixed

    return dataset.map(_mix, num_parallel_calls=tf.data.AUTOTUNE)


def _cosine_lr(base_lr, total_epochs, steps_per_epoch):
    import tensorflow as tf

    return tf.keras.optimizers.schedules.CosineDecay(base_lr, total_epochs * steps_per_epoch, alpha=0.1)


def fine_tune_three_phase(model, train_dataset, validation_dataset):
    """Optional 3-phase fine-tune: Phase A (frozen warm-up) -> Phase B (last 10
    backbone layers) -> Phase C (full backbone), all with focal loss + MixUp
    (RGB only) + cosine-decay LR. NOTE: in the original experiment this
    overfit — the head-only model is the one actually reported / deployed.
    Requires `pip install tensorflow-probability`.
    """
    import tensorflow as tf

    config.FINETUNE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(config.FINETUNE_CHECKPOINT_DIR / "best_ft_epoch_{epoch:02d}_val_auc_{val_auc:.4f}.keras")

    class_weights = compute_class_weights("train")
    steps_per_epoch = int(tf.data.experimental.cardinality(train_dataset).numpy())

    train_ds_mix = _make_mixup(train_dataset, config.FINE_TUNE_MIXUP_ALPHA)
    metric_auc = tf.keras.metrics.AUC(name="auc")

    cb_ckpt = ModelCheckpoint(ckpt_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1)

    backbone = model.get_layer("efficientnetv2-s")

    # --- Phase A: frozen warm-up, MixUp ON ---
    for layer in backbone.layers:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(config.FT_PHASE_A_LR, weight_decay=config.FT_PHASE_A_WEIGHT_DECAY),
        loss=_focal_loss,
        metrics=[metric_auc],
    )
    print(f"--- Fine-tune Phase A: {config.FT_PHASE_A_EPOCHS} epochs, backbone frozen, MixUp ON ---")
    history_a = model.fit(
        train_ds_mix,
        validation_data=validation_dataset,
        epochs=config.FT_PHASE_A_EPOCHS,
        class_weight=class_weights,
        callbacks=[cb_ckpt],
    )

    # --- Phase B: unfreeze last N backbone layers (excluding BatchNorm), MixUp ON ---
    for layer in backbone.layers[-config.FT_PHASE_B_UNFREEZE_LAST_N_LAYERS:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    cb_stop_b = EarlyStopping(
        monitor="val_auc", mode="max", patience=config.FT_PHASE_B_EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=_cosine_lr(config.FT_PHASE_B_BASE_LR, config.FT_PHASE_B_EPOCHS, steps_per_epoch),
            weight_decay=config.FT_PHASE_B_WEIGHT_DECAY,
        ),
        loss=_focal_loss,
        metrics=[metric_auc],
    )
    print(f"--- Fine-tune Phase B: {config.FT_PHASE_B_EPOCHS} epochs, "
          f"last {config.FT_PHASE_B_UNFREEZE_LAST_N_LAYERS} backbone layers unfrozen, MixUp ON ---")
    history_b = model.fit(
        train_ds_mix,
        validation_data=validation_dataset,
        epochs=config.FT_PHASE_B_EPOCHS,
        class_weight=class_weights,
        callbacks=[cb_ckpt, cb_stop_b],
    )

    # --- Phase C: unfreeze full backbone (excluding BatchNorm), MixUp OFF ---
    for layer in backbone.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    cb_stop_c = EarlyStopping(
        monitor="val_auc", mode="max", patience=config.FT_PHASE_C_EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=_cosine_lr(config.FT_PHASE_C_BASE_LR, config.FT_PHASE_C_EPOCHS, steps_per_epoch),
            weight_decay=config.FT_PHASE_C_WEIGHT_DECAY,
        ),
        loss=_focal_loss,
        metrics=[metric_auc],
    )
    print(f"--- Fine-tune Phase C: {config.FT_PHASE_C_EPOCHS} epochs, full backbone unfrozen, MixUp OFF ---")
    history_c = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.FT_PHASE_C_EPOCHS,
        class_weight=class_weights,
        callbacks=[cb_ckpt, cb_stop_c],
    )

    print(f"Fine-tuning done. Best checkpoints saved under: {config.FINETUNE_CHECKPOINT_DIR}")
    return history_a, history_b, history_c


def main():
    from .model_architecture import SRMLayer

    parser = argparse.ArgumentParser(description="Train the RGB+ELA+SRM shared-EfficientNetV2S forgery detector.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_HEAD)
    parser.add_argument("--fine-tune", action="store_true", help="Also run the (overfit-prone) 3-phase fine-tune.")
    parser.add_argument("--plot", action="store_true", help="Show/save training curves after training.")
    args = parser.parse_args()

    set_seed(config.SEED)
    train_dataset, validation_dataset, class_names = get_tf_datasets()

    model = build_rgb_ela_srm_effnetv2s_model()
    model = compile_head_model(model)
    model.summary()

    history = train_head(model, train_dataset, validation_dataset, epochs=args.epochs)
    if args.plot:
        plot_training_history(history, save_path=str(config.RESULTS_DIR / "head_training_curves.png"))

    if args.fine_tune:
        import tensorflow as tf

        model = tf.keras.models.load_model(str(config.HEAD_MODEL_PATH), custom_objects={"SRMLayer": SRMLayer})
        fine_tune_three_phase(model, train_dataset, validation_dataset)


if __name__ == "__main__":
    main()

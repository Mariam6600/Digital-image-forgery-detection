"""Model architecture for Methodology 1: dual-input (RGB + ELA) ResNet50V2 with late fusion."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from . import config


def build_robust_dual_input_model(
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
    num_classes: int = 1,
) -> Model:
    """Build the dual-stream (RGB + ELA) forgery classifier.

    Two independent, frozen ResNet50V2 backbones extract features from the
    RGB image and its ELA map respectively; the pooled features are
    concatenated and passed through a regularized dense head.

    - RGB stream: heavier augmentation (flips, rotation, zoom, translation,
      contrast, brightness) since natural photos tolerate more distortion.
    - ELA stream: only a horizontal flip — geometric/photometric augmentation
      would corrupt the subtle compression-error signal ELA relies on.
    """
    import tensorflow as tf

    # --- RGB stream ---
    rgb_input = Input(shape=input_shape, name="rgb_input")
    rgb_aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.25),
            layers.RandomZoom(height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3)),
            layers.RandomTranslation(height_factor=0.15, width_factor=0.15),
            layers.RandomContrast(factor=0.3),
            layers.RandomBrightness(factor=0.3),
        ],
        name="rgb_strong_augmentation",
    )(rgb_input)
    rgb_preprocessed = tf.keras.applications.resnet_v2.preprocess_input(rgb_aug)

    base_model_rgb = ResNet50V2(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg", name="resnet50v2_rgb_base"
    )
    base_model_rgb.trainable = False
    rgb_features = base_model_rgb(rgb_preprocessed, training=False)

    # --- ELA stream ---
    ela_input = Input(shape=input_shape, name="ela_input")
    ela_aug = keras.Sequential([layers.RandomFlip("horizontal")], name="ela_augmentation")(ela_input)
    ela_rescaled = layers.Rescaling(1.0 / 255.0, name="ela_rescaling")(ela_aug)

    base_model_ela = ResNet50V2(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg", name="resnet50v2_ela_base"
    )
    base_model_ela.trainable = False
    ela_features = base_model_ela(ela_rescaled, training=False)

    # --- Fusion + classification head ---
    fused = concatenate([rgb_features, ela_features], name="concatenated_features")

    x = layers.Dense(512, activation="relu", kernel_regularizer=l2(0.0015), name="dense_intermediate_1")(fused)
    x = layers.BatchNormalization(name="batch_norm_intermediate_1")(x)
    x = layers.Dropout(0.65, name="dropout_intermediate_1")(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(0.0015), name="dense_intermediate_2")(x)
    x = layers.BatchNormalization(name="batch_norm_intermediate_2")(x)
    x = layers.Dropout(0.55, name="dropout_intermediate_2")(x)

    outputs = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)

    return Model(inputs=[rgb_input, ela_input], outputs=outputs, name="RobustDualInputHeadModel")

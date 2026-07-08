"""Model architecture for Methodology 2: single-input ResNet101V2 on ELA images."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from . import config


def build_ela_resnet101v2_model(
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
    num_classes: int = 1,
) -> Model:
    """Build the ELA-only forgery classifier.

    Architecture: ELA image -> light augmentation -> rescale [0,1] -> ResNet-V2
    preprocessing [-1,1] -> frozen ImageNet ResNet101V2 backbone (avg pooling)
    -> Dense(512) -> Dense(256) -> sigmoid.

    The backbone starts frozen (`trainable=False`); use
    `src.training.unfreeze_for_fine_tuning` to selectively unfreeze the last
    block for an optional fine-tuning stage.
    """
    ela_input = Input(shape=input_shape, name="ela_input")

    # Minimal augmentation: ELA maps encode subtle statistical artifacts, so we
    # avoid geometric transforms (rotation/zoom) that could distort them.
    x = keras.Sequential([layers.RandomFlip("horizontal")], name="ela_augmentation")(ela_input)
    x = layers.Rescaling(1.0 / 255.0, name="ela_rescaling_to_0_1")(x)

    import tensorflow as tf

    x = tf.keras.applications.resnet_v2.preprocess_input(x)

    base_model = ResNet101V2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
        name="resnet101v2_ela_base",
    )
    base_model.trainable = False
    x = base_model(x, training=False)

    x = layers.Dense(512, activation="relu", kernel_regularizer=l2(0.0015), name="dense_intermediate_1")(x)
    x = layers.BatchNormalization(name="batch_norm_intermediate_1")(x)
    x = layers.Dropout(0.65, name="dropout_intermediate_1")(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=l2(0.0015), name="dense_intermediate_2")(x)
    x = layers.BatchNormalization(name="batch_norm_intermediate_2")(x)
    x = layers.Dropout(0.55, name="dropout_intermediate_2")(x)

    outputs = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)

    return Model(inputs=ela_input, outputs=outputs, name="ELA_Only_ResNet101V2_Model")

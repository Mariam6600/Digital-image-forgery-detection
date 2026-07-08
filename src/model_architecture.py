"""
Model architecture for Methodology 3: RGB + ELA + SRM features fused through a
single, weight-shared EfficientNetV2S backbone.

- RGB and ELA images are both fed through the *same* EfficientNetV2S instance
  (weight sharing), each followed by its own BatchNormalization.
- SRM (Steganalysis Rich Model) residual features are computed directly from
  the RGB image with a small, fixed (non-trainable) high-pass filter bank,
  then globally average-pooled — no backbone is applied to this branch.
- The three feature vectors are concatenated and passed through a dense head.

NOTE: unlike Methodologies 1 & 2, this pipeline feeds images in [0, 1] range
without ResNetV2/EfficientNet-style `preprocess_input`. This matches exactly
what produced the reported results in the original experiment; it is flagged
here for transparency rather than "fixed", since changing it would no longer
reproduce the original numbers. See README "Known quirk" note.
"""
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetV2S

from . import config


class SRMLayer(layers.Layer):
    """Fixed 3-filter SRM high-pass bank (3x3), non-trainable.

    Highlights local pixel-value inconsistencies that raw RGB/ELA views don't
    directly expose — a classic steganalysis/forensics feature.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        k1 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        k2 = [[0, 0, 0], [0, -1, 0], [0, 0, 0]]
        k3 = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]

        kernels = tf.constant([k1, k2, k3], dtype=tf.float32)  # (3, 3, 3)
        kernels = tf.reshape(kernels, (3, 3, 1, 3))  # (kH, kW, inC, outC)
        self.kernels = tf.repeat(kernels, 3, axis=2)  # RGB -> inC=3

    def call(self, x):
        x = tf.nn.conv2d(x, self.kernels, strides=1, padding="SAME")
        return tf.math.abs(x)

    def compute_output_shape(self, input_shape):
        return input_shape  # still (None, H, W, 3)

    def get_config(self):
        return super().get_config()


def build_rgb_ela_srm_effnetv2s_model(
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
) -> Model:
    """Build the RGB + ELA + SRM forgery classifier.

    Returns a Keras Model with two named inputs: 'rgb_input' and 'ela_input'
    (SRM is derived internally from the RGB input, not a separate input).
    """
    rgb_in = layers.Input(input_shape, name="rgb_input")
    ela_in = layers.Input(input_shape, name="ela_input")

    # --- SRM branch (derived from RGB) ---
    srm_in = SRMLayer(name="srm_from_rgb")(rgb_in)

    # --- Shared EfficientNetV2S backbone (weight-tied across RGB and ELA) ---
    backbone = EfficientNetV2S(include_top=False, weights="imagenet", pooling="avg", name="efficientnetv2-s")

    def backbone_branch(x, name):
        x = backbone(x)
        x = layers.BatchNormalization(name=f"bn_{name}")(x)
        return x

    feat_rgb = backbone_branch(rgb_in, "rgb")
    feat_ela = backbone_branch(ela_in, "ela")
    feat_srm = layers.GlobalAveragePooling2D(name="gap_srm")(srm_in)

    # --- Fusion + classification head ---
    x = layers.Concatenate(name="concat_feats")([feat_rgb, feat_ela, feat_srm])
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = Model(inputs=[rgb_in, ela_in], outputs=output, name="RGB_ELA_SRM_EffNetV2S")
    return model

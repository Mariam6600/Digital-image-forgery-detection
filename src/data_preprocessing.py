"""
Data preparation pipeline for Methodology 1 (Dual-Input RGB + ELA, ResNet50V2).

Pipeline stages (each is idempotent — safe to re-run, will skip work that is
already done unless `force=True` is passed):

    1. download_casia2()          -> downloads CASIA 2.0 via kagglehub
    2. stage_authentic_forged()   -> finds Au/ and Tp/ folders, copies into
                                      data/staged/{authentic,forged}
    3. train_val_split()          -> splits into data/train_val_split/{train,validation} (RGB)
    4. convert_tif_to_png()       -> converts remaining .tif/.tiff files to .png in place
    5. build_ela_dataset()        -> generates ELA images mirroring the RGB split
                                      into data/ela_dataset/{train,validation}
    6. get_tf_datasets()          -> zips the RGB and ELA folders into a single
                                      tf.data.Dataset yielding ((rgb, ela), label)

Run the whole pipeline with:  python -m src.data_preprocessing
"""
import argparse
import os
import shutil

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config
from .utils import generate_ela_image, set_seed

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def download_casia2() -> str:
    """Download the CASIA 2.0 dataset via kagglehub and return the local path."""
    import kagglehub

    print(f"Downloading '{config.KAGGLE_DATASET_SLUG}' from KaggleHub...")
    path = kagglehub.dataset_download(config.KAGGLE_DATASET_SLUG)
    print(f"Dataset available at: {path}")
    return path


def _locate_au_tp_dirs(base_path: str):
    """Recursively find the CASIA2 'Au' (authentic) and 'Tp' (tampered) folders."""

    def _search(root_path):
        au, tp = None, None
        for root, dirs, _files in os.walk(root_path):
            if "Au" in dirs:
                candidate = os.path.join(root, "Au")
                if any(f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(candidate)):
                    au = candidate
            if "Tp" in dirs:
                candidate = os.path.join(root, "Tp")
                if any(f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(candidate)):
                    tp = candidate
            if au and tp:
                break
        return au, tp

    au_dir, tp_dir = _search(base_path)
    if not (au_dir and tp_dir):
        subfolders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        if len(subfolders) == 1:
            au_dir, tp_dir = _search(os.path.join(base_path, subfolders[0]))

    if not (au_dir and tp_dir):
        raise FileNotFoundError(
            f"Could not locate 'Au' and 'Tp' folders under {base_path}. "
            "Inspect the downloaded dataset manually and adjust _locate_au_tp_dirs if needed."
        )
    return au_dir, tp_dir


# --------------------------------------------------------------------------- #
# 2. Stage into authentic/forged
# --------------------------------------------------------------------------- #
def _copy_files(src_dir, dst_dir, extensions=IMAGE_EXTENSIONS) -> int:
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(extensions):
            shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))
            count += 1
    return count


def stage_authentic_forged(force: bool = False) -> str:
    """Copy the raw Au/Tp images into data/staged/{authentic,forged}."""
    if config.STAGED_DIR.exists() and not force:
        print(f"[skip] {config.STAGED_DIR} already exists (use force=True to redo).")
        return str(config.STAGED_DIR)

    raw_path = download_casia2()
    au_dir, tp_dir = _locate_au_tp_dirs(raw_path)
    print(f"Authentic images: {au_dir}\nTampered images:  {tp_dir}")

    if config.STAGED_DIR.exists():
        shutil.rmtree(config.STAGED_DIR)

    n_auth = _copy_files(au_dir, config.STAGED_DIR / "authentic")
    n_forged = _copy_files(tp_dir, config.STAGED_DIR / "forged")
    print(f"Staged {n_auth} authentic and {n_forged} forged images -> {config.STAGED_DIR}")

    if n_auth == 0 or n_forged == 0:
        raise RuntimeError("No images were staged — check the source directories.")
    if abs(n_auth - n_forged) > 0.1 * (n_auth + n_forged):
        print("WARNING: classes look imbalanced (>10% difference in counts).")
    return str(config.STAGED_DIR)


# --------------------------------------------------------------------------- #
# 3. Train / validation split (RGB)
# --------------------------------------------------------------------------- #
def _split_and_copy(source_dir, train_dir, val_dir, val_ratio, seed):
    filenames = os.listdir(source_dir)
    rng = np.random.RandomState(seed)
    rng.shuffle(filenames)
    train_files, val_files = train_test_split(
        filenames, test_size=val_ratio, random_state=seed, shuffle=True
    )
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(val_dir, f))
    return len(train_files), len(val_files)


def train_val_split(force: bool = False) -> str:
    """Split data/staged/{authentic,forged} into train_val_split/{train,validation}."""
    if config.SPLIT_DIR.exists() and not force:
        print(f"[skip] {config.SPLIT_DIR} already exists (use force=True to redo).")
        return str(config.SPLIT_DIR)

    if config.SPLIT_DIR.exists():
        shutil.rmtree(config.SPLIT_DIR)

    for cls in config.CLASS_NAMES:
        src = config.STAGED_DIR / cls
        n_train, n_val = _split_and_copy(
            src,
            config.SPLIT_DIR / "train" / cls,
            config.SPLIT_DIR / "validation" / cls,
            config.VAL_SPLIT_RATIO,
            config.SEED,
        )
        print(f"'{cls}': {n_train} train / {n_val} validation")
    return str(config.SPLIT_DIR)


# --------------------------------------------------------------------------- #
# 4. TIF -> PNG conversion
# --------------------------------------------------------------------------- #
def convert_tif_to_png() -> None:
    """CASIA2 ships some tampered images as .tif; convert them to .png in place
    (tf.keras.utils.image_dataset_from_directory does not read TIFF)."""
    folders = [
        config.SPLIT_DIR / "train" / "authentic",
        config.SPLIT_DIR / "train" / "forged",
        config.SPLIT_DIR / "validation" / "authentic",
        config.SPLIT_DIR / "validation" / "forged",
    ]
    converted = 0
    for folder in folders:
        if not folder.exists():
            continue
        for path in list(folder.glob("*.tif")) + list(folder.glob("*.TIF")) + \
                list(folder.glob("*.tiff")) + list(folder.glob("*.TIFF")):
            try:
                with Image.open(path) as img:
                    png_path = path.with_suffix(".png")
                    img.save(png_path, "PNG")
                os.remove(path)
                converted += 1
            except Exception as exc:  # noqa: BLE001
                print(f"Could not convert {path}: {exc}")
    print(f"Converted {converted} TIF/TIFF files to PNG.")


# --------------------------------------------------------------------------- #
# 5. ELA dataset generation (mirrors the RGB split)
# --------------------------------------------------------------------------- #
def build_ela_dataset(force: bool = False) -> str:
    """Generate ELA images mirroring data/train_val_split/ into data/ela_dataset/."""
    if config.ELA_DIR.exists() and not force:
        print(f"[skip] {config.ELA_DIR} already exists (use force=True to redo).")
        return str(config.ELA_DIR)

    if config.ELA_DIR.exists():
        shutil.rmtree(config.ELA_DIR)

    total_in, total_out = 0, 0
    for split in ("train", "validation"):
        for cls in config.CLASS_NAMES:
            src_dir = config.SPLIT_DIR / split / cls
            dst_dir = config.ELA_DIR / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not src_dir.exists():
                print(f"WARNING: missing source folder {src_dir}, skipping.")
                continue
            for filename in os.listdir(src_dir):
                src_path = src_dir / filename
                if not (src_path.is_file() and filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))):
                    continue
                total_in += 1
                ela_img = generate_ela_image(str(src_path), quality=config.ELA_QUALITY, scale_factor=config.ELA_SCALE)
                if ela_img is not None:
                    ela_img.save(dst_dir / (os.path.splitext(filename)[0] + ".png"))
                    total_out += 1
    print(f"ELA generation done: {total_out}/{total_in} images -> {config.ELA_DIR}")
    return str(config.ELA_DIR)


# --------------------------------------------------------------------------- #
# 6. tf.data.Dataset loading — RGB and ELA zipped into a dual-input dataset
# --------------------------------------------------------------------------- #
def get_tf_datasets():
    """Load the RGB and ELA train/validation folders and zip them into a dual-input
    tf.data.Dataset yielding ((rgb_batch, ela_batch), label_batch).

    Returns:
        (train_dataset, validation_dataset, class_names)
    """
    import tensorflow as tf

    rgb_train_dir = config.SPLIT_DIR / "train"
    rgb_val_dir = config.SPLIT_DIR / "validation"
    ela_train_dir = config.ELA_DIR / "train"
    ela_val_dir = config.ELA_DIR / "validation"
    for d in (rgb_train_dir, rgb_val_dir, ela_train_dir, ela_val_dir):
        if not d.exists() or not any(d.iterdir()):
            raise FileNotFoundError(f"{d} is missing or empty — run the data pipeline first.")

    def _load(directory, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="binary",
            image_size=config.IMAGE_SIZE,
            interpolation="nearest",
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            seed=config.SEED if shuffle else None,
        )

    print("--- Loading RGB datasets ---")
    rgb_train_raw = _load(rgb_train_dir, shuffle=True)
    class_names = rgb_train_raw.class_names
    rgb_val_raw = _load(rgb_val_dir, shuffle=False)

    print("--- Loading ELA datasets ---")
    ela_train_raw = _load(ela_train_dir, shuffle=True)  # same seed -> same shuffle order as RGB
    if ela_train_raw.class_names != class_names:
        print(f"WARNING: ELA class names {ela_train_raw.class_names} != RGB class names {class_names}. "
              "RGB/ELA pairing may be wrong!")
    ela_val_raw = _load(ela_val_dir, shuffle=False)

    def _combine(rgb_data, ela_data):
        rgb_images, rgb_labels = rgb_data
        ela_images, _ela_labels = ela_data
        return (rgb_images, ela_images), rgb_labels

    autotune = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.zip((rgb_train_raw, ela_train_raw)).map(_combine, num_parallel_calls=autotune)
    train_dataset = train_dataset.prefetch(buffer_size=autotune)

    validation_dataset = tf.data.Dataset.zip((rgb_val_raw, ela_val_raw)).map(_combine, num_parallel_calls=autotune)
    validation_dataset = validation_dataset.prefetch(buffer_size=autotune)

    print(f"Classes: {class_names}")
    print(f"Train batches: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Validation batches: {tf.data.experimental.cardinality(validation_dataset).numpy()}")
    return train_dataset, validation_dataset, class_names


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def prepare_all(force: bool = False) -> None:
    set_seed(config.SEED)
    stage_authentic_forged(force=force)
    train_val_split(force=force)
    convert_tif_to_png()
    build_ela_dataset(force=force)
    print("\nData pipeline complete. Folders ready under:", config.DATA_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Prepare CASIA2 + ELA dataset for Methodology 1 (dual-input).")
    parser.add_argument("--force", action="store_true", help="Re-run every stage even if outputs already exist.")
    args = parser.parse_args()
    prepare_all(force=args.force)


if __name__ == "__main__":
    main()

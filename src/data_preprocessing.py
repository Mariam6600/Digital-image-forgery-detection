"""
Data preparation pipeline for Methodology 3 (RGB + ELA + SRM, shared EfficientNetV2S).

Pipeline stages (each is idempotent — safe to re-run, will skip work that is
already done unless `force=True` is passed):

    1. download_casia2()          -> downloads CASIA 2.0 via kagglehub
    2. stage_authentic_forged()   -> finds Au/ and Tp/ folders, copies into
                                      data/staged/{authentic,forged}
    3. train_val_split()          -> splits into data/train_val_split/{train,validation} (RGB)
    4. convert_tif_to_png()       -> converts remaining .tif/.tiff files to .png in place
    5. build_ela_dataset()        -> generates ELA images mirroring the RGB split
                                      into data/ela_dataset/{train,validation}
    6. get_tf_datasets()          -> pairs RGB/ELA files by filename stem and builds
                                      a tf.data.Dataset yielding
                                      ({'rgb_input': rgb, 'ela_input': ela}, label)
                                      with both images rescaled to [0,1]
                                      (no ResNetV2-style preprocessing, no
                                      augmentation — matches the original experiment).

Run the whole pipeline with:  python -m src.data_preprocessing
"""
import argparse
import glob
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
    """CASIA2 ships some tampered images as .tif; convert them to .png in place."""
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
# 6. tf.data.Dataset loading — pair RGB/ELA files by filename stem
# --------------------------------------------------------------------------- #
EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _collect_pairs(rgb_root, ela_root):
    """Match every RGB file to its corresponding ELA file by class + filename stem."""
    rgb_paths, ela_paths, labels = [], [], []
    for label, cls in enumerate(sorted(os.listdir(rgb_root))):  # authentic/forged
        rgb_cls_dir = os.path.join(rgb_root, cls)
        ela_cls_dir = os.path.join(ela_root, cls)
        for rgb_fp in glob.glob(rgb_cls_dir + "/**/*", recursive=True):
            if os.path.splitext(rgb_fp)[1].lower() not in EXTS:
                continue
            stem = os.path.splitext(os.path.basename(rgb_fp))[0]
            ela_fp = os.path.join(ela_cls_dir, stem + ".png")  # ELA always saved as .png
            if os.path.exists(ela_fp):
                rgb_paths.append(rgb_fp)
                ela_paths.append(ela_fp)
                labels.append(label)
    return rgb_paths, ela_paths, labels


def get_tf_datasets():
    """Build the dict-input dataset: ({'rgb_input': rgb, 'ela_input': ela}, label).

    Both images are resized to IMAGE_SIZE and rescaled to [0, 1] — this
    matches the original experiment exactly (no ResNetV2/EfficientNet
    preprocess_input call, no augmentation layers).

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

    def _load_imgs(rgb_fp, ela_fp, label):
        rgb = tf.io.read_file(rgb_fp)
        rgb = tf.io.decode_image(rgb, channels=3, expand_animations=False)
        ela = tf.io.read_file(ela_fp)
        ela = tf.io.decode_image(ela, channels=3, expand_animations=False)
        rgb = tf.image.resize(rgb, config.IMAGE_SIZE)
        ela = tf.image.resize(ela, config.IMAGE_SIZE)
        rgb = tf.cast(rgb, tf.float32) / 255.0
        ela = tf.cast(ela, tf.float32) / 255.0
        return {"rgb_input": rgb, "ela_input": ela}, tf.cast(label, tf.float32)

    def _make_dataset(rgb_root, ela_root, shuffle):
        rgb_p, ela_p, y = _collect_pairs(str(rgb_root), str(ela_root))
        if not rgb_p:
            raise RuntimeError(f"No matching RGB/ELA pairs found under {rgb_root} / {ela_root}.")
        ds = tf.data.Dataset.from_tensor_slices((rgb_p, ela_p, y))
        if shuffle:
            ds = ds.shuffle(len(rgb_p), seed=config.SEED)
        ds = ds.map(_load_imgs, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_dataset = _make_dataset(rgb_train_dir, ela_train_dir, shuffle=True)
    validation_dataset = _make_dataset(rgb_val_dir, ela_val_dir, shuffle=False)
    class_names = config.CLASS_NAMES

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
    parser = argparse.ArgumentParser(description="Prepare CASIA2 + ELA dataset for Methodology 3 (RGB+ELA+SRM).")
    parser.add_argument("--force", action="store_true", help="Re-run every stage even if outputs already exist.")
    args = parser.parse_args()
    prepare_all(force=args.force)


if __name__ == "__main__":
    main()

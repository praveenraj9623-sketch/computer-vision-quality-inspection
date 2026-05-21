from pathlib import Path
import tensorflow as tf
from src.config import IMAGE_SIZE, BATCH_SIZE, SEED


def validate_dataset_dir(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")
    return train_dir, test_dir


def load_train_validation_datasets(dataset_dir: str, validation_split: float = 0.2):
    train_dir, _ = validate_dataset_dir(dataset_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=validation_split, subset="training", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="binary")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=validation_split, subset="validation", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="binary")
    return train_ds, val_ds


def load_test_dataset(dataset_dir: str):
    _, test_dir = validate_dataset_dir(dataset_dir)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False, label_mode="binary")
    return test_ds


def optimize_dataset(ds):
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

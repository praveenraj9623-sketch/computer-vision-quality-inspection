import argparse
from pathlib import Path
import tensorflow as tf
from src.data_utils import load_train_validation_datasets, load_test_dataset, optimize_dataset
from src.model_builder import build_cnn_model, build_mobilenet_model
from src.reporting import evaluate_binary_predictions, save_metrics
from src.config import DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train quality inspection model.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="mobilenet", choices=["mobilenet", "cnn"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    return parser.parse_args()


def collect_predictions(model, dataset):
    y_true, y_prob = [], []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0).reshape(-1)
        y_prob.extend(probs.tolist())
        y_true.extend(labels.numpy().reshape(-1).astype(int).tolist())
    return y_true, y_prob


def main():
    args = parse_args()
    train_ds, val_ds = load_train_validation_datasets(args.dataset_dir)
    test_ds = load_test_dataset(args.dataset_dir)
    train_ds = optimize_dataset(train_ds)
    val_ds = optimize_dataset(val_ds)
    test_ds = optimize_dataset(test_ds)
    model = build_mobilenet_model(train_base=False) if args.model_type == "mobilenet" else build_cnn_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=args.model_path, monitor="val_loss", save_best_only=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    y_true, y_prob = collect_predictions(model, test_ds)
    metrics, cm, report_df = evaluate_binary_predictions(y_true, y_prob)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, DEFAULT_OUTPUT_DIR / "training_metrics.json")
    report_df.to_csv(DEFAULT_OUTPUT_DIR / "classification_report.csv", index=False)
    print("Training completed.")
    print(f"Model saved to: {model_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

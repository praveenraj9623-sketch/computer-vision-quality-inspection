import argparse
from pathlib import Path
import tensorflow as tf
from src.data_utils import load_test_dataset, optimize_dataset
from src.reporting import evaluate_binary_predictions, save_metrics
from src.visualization import plot_confusion_matrix
from src.config import DEFAULT_OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained quality inspection model.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
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
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    test_ds = optimize_dataset(load_test_dataset(args.dataset_dir))
    y_true, y_prob = collect_predictions(model, test_ds)
    metrics, cm, report_df = evaluate_binary_predictions(y_true, y_prob)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, DEFAULT_OUTPUT_DIR / "evaluation_metrics.json")
    report_df.to_csv(DEFAULT_OUTPUT_DIR / "evaluation_classification_report.csv", index=False)
    fig = plot_confusion_matrix(cm)
    fig.savefig(DEFAULT_OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Evaluation completed.")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrix(cm, class_names=None):
    if class_names is None:
        class_names = ["Defective", "OK"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    return fig


def dataframe_from_metrics(metrics: dict):
    return pd.DataFrame([metrics])

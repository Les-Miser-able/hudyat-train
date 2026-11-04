from __future__ import annotations

import numpy as np


def classification_report(actual, predicted, classes: list[str]) -> dict:
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for truth, guess in zip(actual, predicted):
        matrix[int(truth), int(guess)] += 1
    support, guesses, tp = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    precision = np.divide(tp, guesses, out=np.zeros(len(classes), float), where=guesses > 0)
    recall = np.divide(tp, support, out=np.zeros(len(classes), float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(classes), float), where=(precision + recall) > 0)
    accuracy = float(tp.sum() / matrix.sum()) if matrix.sum() else 0.0
    low = [name for name, score in zip(classes, recall) if score < 0.7]
    return {"accuracy": accuracy, "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(), "samples": int(matrix.sum()),
            "per_class": [{"label": name, "support": int(support[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i])} for i, name in enumerate(classes)],
            "benchmark": {"accuracy_at_least_85": accuracy >= 0.85, "macro_f1_at_least_85": float(f1.mean()) >= 0.85,
                          "signs_below_70_recall": low, "classification_pass": accuracy >= 0.85 and float(f1.mean()) >= 0.85 and not low,
                          "deployment_response_time": "not measured; requires the HUDYAT browser on the deployment device"}}


def plot_reports(run, history, report, classes):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, axis in zip(("loss", "accuracy"), axes):
        epochs = range(1, len(history[metric]) + 1)
        marker = "." if len(history[metric]) < 3 else None
        axis.plot(epochs, history[metric], label="Training", marker=marker)
        axis.plot(epochs, history["val_" + metric], label="Validation", marker=marker)
        axis.set(xlabel="Epoch", ylabel=metric.title())
        if metric == "accuracy":
            axis.set_ylim(0, 1)
        axis.legend()
    fig.tight_layout()
    fig.savefig(run / "learning-curves.png", dpi=150)
    plt.close(fig)
    size = max(6, len(classes) * 0.35)
    fig, axis = plt.subplots(figsize=(size, size))
    matrix = np.asarray(report["confusion_matrix"])
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
             xlabel="Predicted", ylabel="Actual")
    plt.setp(axis.get_xticklabels(), rotation=90)
    for i in range(len(classes)):
        for j in range(len(classes)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(run / "confusion-matrix.png", dpi=150)
    plt.close(fig)

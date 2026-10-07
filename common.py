"""Shared reproducibility and reporting helpers, distributed with each project."""

import hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
)

SEED = 42
plt.rcParams.update(
    {
        "figure.figsize": (9, 5),
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    }
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=lambda x: x.item() if hasattr(x, "item") else str(x),
        ),
        encoding="utf8",
    )


def evaluate(y, pred, prob=None):
    d = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "weighted_f1": f1_score(y, pred, average="weighted"),
    }
    if prob is not None and len(np.unique(y)) == 2:
        d.update(
            roc_auc=roc_auc_score(y, prob),
            average_precision=average_precision_score(y, prob),
        )
    return {k: float(v) for k, v in d.items()}


def classification_artifacts(y, pred, out, name="test", labels=None):
    out = Path(out)
    save_json(
        out / f"{name}_classification_report.json",
        classification_report(y, pred, output_dict=True, zero_division=0),
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, ax=ax, colorbar=False, display_labels=labels, cmap="Blues"
    )
    ax.set_title("Held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(out / f"{name}_confusion_matrix.png")
    plt.close(fig)


def compare_plot(table, metric, out):
    fig, ax = plt.subplots()
    t = table.sort_values(metric)
    ax.barh(t["model"], t[metric], color="#435fa5")
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title("Validation model comparison")
    for i, x in enumerate(t[metric]):
        ax.text(min(x + 0.01, 0.94), i, f"{x:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def versions():
    import sklearn, scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "seed": SEED,
    }


def accuracy_interval(y, pred):
    # Wilson 95% interval for held-out accuracy; no distributional claim across datasets.
    n = len(y)
    p = float(np.mean(np.asarray(y) == np.asarray(pred)))
    z = 1.96
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(center - half), float(center + half)]

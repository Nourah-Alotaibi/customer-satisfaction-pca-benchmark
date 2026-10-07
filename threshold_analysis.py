"""Select an illustrative decision threshold from group-aware development OOF scores.

This is a follow-up analysis on the existing test set, not a fresh independent test.
The threshold rule and its value do not use test labels.
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from train import split_indices
from common import evaluate, save_json, classification_artifacts, plt, SEED


def select_f1_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = (
        2
        * precision[:-1]
        * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    # Fixed tie rule: select the first maximum; no test labels enter this function.
    best = int(np.argmax(f1))
    return float(thresholds[best]), precision, recall, f1


def run(data, model_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data)
    y = df.pop("TARGET").astype(int)
    X = df.drop(columns=["ID"], errors="ignore").astype("float32")
    dev, _, _, test, groups = split_indices(X, y)
    model = joblib.load(model_path)
    oof = np.zeros(len(dev))
    folds = []
    for fold, (fit, score) in enumerate(
        StratifiedGroupKFold(4, shuffle=True, random_state=SEED + 2).split(
            X.iloc[dev], y.iloc[dev], groups[dev]
        ),
        1,
    ):
        assert not set(groups[dev[fit]]) & set(groups[dev[score]])
        candidate = clone(model)
        candidate.fit(X.iloc[dev[fit]], y.iloc[dev[fit]])
        oof[score] = candidate.predict_proba(X.iloc[dev[score]])[:, 1]
        folds.append({"fold": fold, "fit_rows": len(fit), "score_rows": len(score)})
        print("Completed group-aware development fold", fold, flush=True)
    threshold, precision, recall, f1 = select_f1_threshold(y.iloc[dev], oof)
    # Fixed trained model and its original held-out split are used only after locking threshold.
    probability = model.predict_proba(X.iloc[test])[:, 1]
    rows = []
    for name, t in [
        ("Default threshold", 0.5),
        ("Development-OOF F1 threshold", threshold),
    ]:
        pred = (probability >= t).astype(int)
        rows.append(
            {
                "operating_point": name,
                "threshold": t,
                **evaluate(y.iloc[test], pred, probability),
                "minority_precision": float(
                    precision_score(y.iloc[test], pred, zero_division=0)
                ),
                "minority_recall": float(recall_score(y.iloc[test], pred)),
                "minority_f1": float(f1_score(y.iloc[test], pred)),
                "flagged_rows": int(pred.sum()),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out / "threshold_test_comparison.csv", index=False)
    classification_artifacts(
        y.iloc[test],
        (probability >= threshold).astype(int),
        out,
        name="threshold_test",
        labels=["Satisfied", "Dissatisfied"],
    )
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color="#6552a5", label="Development OOF predictions")
    i = int(np.argmax(f1))
    ax.scatter(
        recall[i],
        precision[i],
        color="#c2693e",
        s=65,
        label=f"Maximum OOF F1, threshold={threshold:.3f}",
    )
    ax.set(
        xlabel="Dissatisfied-class recall",
        ylabel="Dissatisfied-class precision",
        title="Choosing a threshold without test labels",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "threshold_selection.png")
    plt.close(fig)
    report = {
        "analysis": "Follow-up operating-point analysis; existing test reused, not fresh validation",
        "selection_rule": "Maximum dissatisfied-class F1 on four-fold group-disjoint development OOF scores",
        "threshold": threshold,
        "development_rows": len(dev),
        "folds": folds,
        "oof_minority_f1": float(max(f1)),
        "test_comparison": rows,
        "limitations": [
            "F1 is an illustrative objective, not a validated business-cost policy.",
            "Threshold generalization needs validation on a fresh time period.",
            "Changing threshold trades false positives for missed positives; ROC-AUC is unchanged.",
        ],
    }
    save_json(out / "threshold_analysis.json", report)
    save_json(
        Path(model_path).parent / "operating_point.json",
        {"threshold": threshold, "selection": "development OOF F1"},
    )
    print(table.to_string(index=False), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="models/model.joblib")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(a.data, a.model, a.out)

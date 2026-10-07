"""Reproducible Santander benchmark. No test-based model selection."""

import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from common import *


def split_indices(X, y):
    """Keep identical predictor rows together across all three partitions."""
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    dev, test = next(
        StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups)
    )
    tr, va = next(
        StratifiedGroupKFold(4, shuffle=True, random_state=SEED + 1).split(
            X.iloc[dev], y.iloc[dev], groups[dev]
        )
    )
    tr, va = dev[tr], dev[va]
    assert not set(groups[tr]) & set(groups[va]) and not set(groups[dev]) & set(
        groups[test]
    )
    return dev, tr, va, test, groups


def run(data, out):
    out = Path(out)
    out.mkdir(exist_ok=True, parents=True)
    start = time.time()
    df = pd.read_csv(data)
    y = df.pop("TARGET").astype(int)
    X = df.drop(columns=["ID"], errors="ignore").astype("float32")
    # Same predictor rows must remain in one partition, even if labels disagree.
    dev, tr, va, test, groups = split_indices(X, y)

    def linear(pca=False):
        steps = [
            SimpleImputer(strategy="median"),
            VarianceThreshold(),
            StandardScaler(),
        ]
        if pca:
            steps.append(
                PCA(n_components=128, svd_solver="randomized", random_state=SEED)
            )
        return make_pipeline(
            *steps, LogisticRegression(C=0.01, max_iter=1500, random_state=SEED)
        )

    candidates = {
        "L2 logistic baseline": linear(),
        "PCA128 + L2": linear(True),
        "HistGradientBoosting leaves15": HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=15,
            learning_rate=0.06,
            l2_regularization=10,
            random_state=SEED,
        ),
        "HistGradientBoosting leaves31": HistGradientBoostingClassifier(
            max_iter=220,
            max_leaf_nodes=31,
            learning_rate=0.05,
            l2_regularization=20,
            random_state=SEED,
        ),
    }
    rows = []
    for name, m in candidates.items():
        print("Fitting", name, flush=True)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        rows.append({"model": name, **evaluate(y.iloc[va], m.predict(X.iloc[va]), p)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "validation_comparison.csv", index=False)
    best = table.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    # Lock selection before evaluating the held-out test set. Refit on development only.
    results = {}
    for name in dict.fromkeys(["L2 logistic baseline", "PCA128 + L2", best]):
        m = candidates[name]
        m.fit(X.iloc[dev], y.iloc[dev])
        pred = m.predict(X.iloc[test])
        prob = m.predict_proba(X.iloc[test])[:, 1]
        results[name] = evaluate(y.iloc[test], pred, prob)
        if name == best:
            classification_artifacts(y.iloc[test], pred, out)
            models = out.parent / "models"
            models.mkdir(exist_ok=True)
            joblib.dump(m, models / "model.joblib")
    compare_plot(table, "roc_auc", out / "validation_comparison.png")
    report = {
        "dataset": "Santander Customer Satisfaction training file",
        "input_sha256": sha256(data),
        "rows": len(X),
        "features": X.shape[1],
        "positive_fraction": float(y.mean()),
        "same_predictor_rows_beyond_first": int(pd.Series(groups).duplicated().sum()),
        "split_sizes": {"train": len(tr), "validation": len(va), "test": len(test)},
        "selection_metric": "validation ROC-AUC",
        "selected_model": best,
        "test_results": results,
        "majority_test_accuracy": float((y.iloc[test] == 0).mean()),
        "legacy_cv_auc": 0.79524,
        "legacy_comparability": "Old CV result is not directly comparable to this group-disjoint held-out test. Same-split baseline comparison is provided.",
        "runtime_seconds": time.time() - start,
        "environment": versions(),
    }
    save_json(out / "metrics.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(a.data, a.out)

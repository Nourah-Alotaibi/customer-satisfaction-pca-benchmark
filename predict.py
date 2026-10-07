"""Score customer records using a locally trained trusted model."""

import argparse
from pathlib import Path
import joblib, pandas as pd


def predict(model_path, input_path, output_path):
    model = joblib.load(model_path)
    data = pd.read_csv(input_path)
    columns = list(model.feature_names_in_)
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError("Missing feature columns: " + ", ".join(sorted(missing)))
    result = pd.DataFrame(
        {
            "unsatisfied_probability": model.predict_proba(
                data[columns].astype("float32")
            )[:, 1]
        }
    )
    if "ID" in data:
        result.insert(0, "ID", data.ID.to_numpy())
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/model.joblib")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="predictions.csv")
    a = p.parse_args()
    predict(a.model, a.input, a.output)

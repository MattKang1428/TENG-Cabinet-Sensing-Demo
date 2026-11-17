"""Predict electrical contact labels from the included example or a feature CSV."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from electrical_inference import FEATURE_NAMES, predict_hierarchical


def predict_csv(input_csv, output_dir, check_example=False):
    started = time.perf_counter()
    bundle = joblib.load(ROOT / "models" / "inference_bundle.pkl")
    if bundle["version"] != "teng-sync-hierarchical-v1" or bundle["features"] != FEATURE_NAMES:
        raise ValueError("Model and feature schema do not match")
    frame = pd.read_csv(input_csv)
    missing = [name for name in FEATURE_NAMES if name not in frame]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    prediction, gate = predict_hierarchical(
        frame[FEATURE_NAMES].to_numpy(dtype=np.float64),
        bundle["gate_pipeline"], bundle["channel_pipeline"],
    )
    actual = pd.DataFrame({
        "sample_id": frame["sample_id"] if "sample_id" in frame
        else [f"row_{i + 1}" for i in range(len(frame))]
    })
    label_column = "label" if "label" in frame else "reference_label" if "reference_label" in frame else None
    if label_column is not None:
        labels = pd.to_numeric(frame[label_column], errors="raise").to_numpy()
        if not np.isin(labels, range(9)).all():
            raise ValueError("Reference labels, when supplied, must be integers 0 through 8")
        actual["reference_label"] = labels.astype(int)
    actual["predicted_gate_label"] = gate
    actual["predicted_label"] = prediction
    if check_example:
        expected = pd.read_csv(ROOT / "expected_output" / "example_predictions.csv")
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    actual.to_csv(output_dir / "predictions.csv", index=False)
    report = {
        "mode": "bundled_example" if check_example else "custom_feature_csv",
        "samples": len(frame),
        "expected_output_checked": check_example,
        "matches_expected_predictions": True if check_example else None,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "model_sha256": hashlib.sha256((ROOT / "models" / "inference_bundle.pkl").read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "joblib")},
    }
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path,
                        help="CSV with the eight named feature columns; sample_id and reference labels are optional")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output",
                        help="Directory for predictions.csv and verification.json")
    args = parser.parse_args()
    result = predict_csv(args.input_csv or ROOT / "demo_data" / "features_example.csv",
                         args.output_dir, check_example=args.input_csv is None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

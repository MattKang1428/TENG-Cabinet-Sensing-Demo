"""Predict optical contact classes and verify the bundled image examples."""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
SPEC = spec_from_file_location("teng_optical_predictor", ROOT / "code" / "predict_8d.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError("Cannot load the bundled optical prediction module")
PREDICTOR = module_from_spec(SPEC)
SPEC.loader.exec_module(PREDICTOR)


def predict_image(path: Path, bundle: dict) -> dict:
    label, scores, count = PREDICTOR.predict(PREDICTOR.read_image(path), bundle)
    return {
        "prediction": label,
        "blue_on_count": count,
        "branch": "multi_light_max_score" if count >= 2 else "lda",
        "blue_scores": scores.tolist(),
    }


def run_demo(output_dir: Path) -> dict:
    """Check image integrity, perform inference and compare with saved outputs."""
    started = time.perf_counter()
    bundle = PREDICTOR.load_model_bundle()
    with (ROOT / "demo_data" / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    expected = json.loads((ROOT / "expected_output" / "predictions.json").read_text(encoding="utf-8"))
    if len(manifest) != len(expected["samples"]):
        raise ValueError("The image manifest and expected output have different lengths")
    samples = []
    for row, reference in zip(manifest, expected["samples"]):
        path = ROOT / row["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Demo image differs from the supplied input: {row['image']}")
        result = {
            "image": row["image"],
            "reference_label": int(row["reference_label"]),
            **predict_image(path, bundle),
        }
        for key in ("image", "reference_label", "prediction", "blue_on_count", "branch"):
            if result[key] != reference[key]:
                raise ValueError(f"Prediction differs from the reference: {row['image']} / {key}")
        np.testing.assert_allclose(result["blue_scores"], reference["blue_scores"], rtol=1e-6, atol=1e-5)
        samples.append(result)
    report = {
        "status": "PASS",
        "model": "StandardScaler + eight-feature Linear Discriminant Analysis",
        "sample_count": len(samples),
        "reference_label_agreement": sum(row["prediction"] == row["reference_label"] for row in samples),
        "selection": expected["selection"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "samples": samples,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "predictions.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: all {len(samples)} image predictions and feature vectors match the expected output.")
    print(f"Output: {destination}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Predict one image instead of running the supplied demo")
    parser.add_argument("--output-dir", type=Path, help="Result directory; default is optical/output")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir else ROOT / "output"
    if args.image is None:
        run_demo(output_dir)
    else:
        result = {"image": args.image.name, **predict_image(args.image, PREDICTOR.load_model_bundle())}
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "prediction.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

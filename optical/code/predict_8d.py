from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pickle

import numpy as np

FEATURE_PATH = Path(__file__).with_name("deployment_features.py")
FEATURE_SPEC = spec_from_file_location("nine_lda3_predict_features", FEATURE_PATH)
if FEATURE_SPEC is None or FEATURE_SPEC.loader is None:
    raise ImportError(f"Could not load deployment features: {FEATURE_PATH}")
FEATURE_MODULE = module_from_spec(FEATURE_SPEC)
FEATURE_SPEC.loader.exec_module(FEATURE_MODULE)
apply_multilight_rule = FEATURE_MODULE.apply_multilight_rule
calculate_all_blue_scores = FEATURE_MODULE.calculate_all_blue_scores
read_image = FEATURE_MODULE.read_image


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model_8d.pkl"


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict:
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle.get("serialization") != "python-pickle":
        raise ValueError("Unexpected model serialization metadata")
    if bundle.get("model_version") != "third-8d-temporal-all-samples-v3":
        raise ValueError("Unsupported 8D model version")
    if bundle.get("feature_implementation_version") != "blue-score-v1":
        raise ValueError("Unsupported blue-score feature implementation")
    if bundle["pipeline"].n_features_in_ != 8:
        raise ValueError("8D model feature count mismatch")
    if bundle["roi_config"]["roi_order"] != list(range(1, 9)):
        raise ValueError("ROI order mismatch")
    return bundle


def predict(image, bundle: dict) -> tuple[int, np.ndarray, int]:
    scores = calculate_all_blue_scores(
        image,
        bundle["roi_config"],
        bundle["threshold_config"],
    )
    threshold = bundle["threshold_config"]["multi_light_thresholds"]
    lda_prediction = int(bundle["pipeline"].predict(scores.reshape(1, -1))[0])
    prediction, on_count, _ = apply_multilight_rule(
        scores,
        lda_prediction,
        bundle["roi_config"]["roi_order"],
        threshold,
    )
    return prediction, scores, on_count

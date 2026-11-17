from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


FEATURE_IMPLEMENTATION_VERSION = "blue-score-v1"
VALID_LABELS = set(range(9))


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def crop_roi(image: np.ndarray, roi: dict) -> np.ndarray:
    height, width = image.shape[:2]
    x = min(max(int(round(float(roi["x"]) * width)), 0), width - 1)
    y = min(max(int(round(float(roi["y"]) * height)), 0), height - 1)
    w = min(max(1, int(round(float(roi["w"]) * width))), width - x)
    h = min(max(1, int(round(float(roi["h"]) * height))), height - y)
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        raise ValueError(f"Empty ROI after scaling: {roi}")
    return crop


def calculate_blue_intensity(roi: np.ndarray, settings: dict) -> float:
    blue, green, red = cv2.split(roi.astype(np.float32))
    blue_excess = blue - (red + green) / 2.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.asarray(settings["blue_hsv_lower"], dtype=np.uint8)
    upper = np.asarray(settings["blue_hsv_upper"], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower, upper) > 0
    bright_mask = blue_mask & (hsv[:, :, 2] >= settings["bright_blue_v_min"])
    positive_excess = np.maximum(blue_excess, 0.0)
    blue_v_mean = float(hsv[:, :, 2][blue_mask].mean()) if blue_mask.any() else 0.0
    return float(
        0.30 * np.percentile(positive_excess, 95)
        + 0.20 * positive_excess.mean()
        + 0.20 * np.percentile(blue, 95)
        + 0.15 * blue_v_mean
        + 0.15 * float(bright_mask.mean()) * 255.0
    )


def calculate_all_blue_scores(
    image: np.ndarray, roi_config: dict, settings: dict
) -> np.ndarray:
    return np.asarray(
        [
            calculate_blue_intensity(
                crop_roi(image, roi_config["rois"][str(number)]), settings
            )
            for number in roi_config["roi_order"]
        ],
        dtype=np.float32,
    )


def apply_multilight_rule(
    scores: np.ndarray,
    lda_prediction: int,
    roi_order: list[int],
    threshold: float | list[float] | dict[str, float],
) -> tuple[int, int, str]:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (8,):
        raise ValueError(f"Expected eight ROI scores, received shape {scores.shape}")
    if roi_order != list(range(1, 9)):
        raise ValueError(f"Unexpected ROI order: {roi_order}")
    if int(lda_prediction) not in VALID_LABELS:
        raise ValueError(f"LDA prediction is outside 0-8: {lda_prediction}")
    if isinstance(threshold, dict):
        thresholds = np.asarray(
            [threshold[str(roi_number)] for roi_number in roi_order],
            dtype=np.float32,
        )
    else:
        thresholds = np.asarray(threshold, dtype=np.float32)
        if thresholds.ndim == 0:
            thresholds = np.full(8, float(thresholds), dtype=np.float32)
    if thresholds.shape != (8,):
        raise ValueError(f"Expected eight ROI thresholds, received shape {thresholds.shape}")
    on_count = int(np.count_nonzero(scores >= thresholds))
    if on_count >= 2:
        final_prediction = int(roi_order[int(np.argmax(scores))])
        return final_prediction, on_count, "multi_light_max_score"
    return int(lda_prediction), on_count, "lda"

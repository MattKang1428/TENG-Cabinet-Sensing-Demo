"""Two-stage electrical classification with the fitted model's own scalers."""

import numpy as np

FEATURE_NAMES = [
    "max", "mean", "peak_ratio", "pvi_seconds", "rise_time_seconds",
    "crest_factor", "fall_half_delay_seconds", "derivative_std",
]


def predict_hierarchical(X, gate, channel):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 8 or not len(X) or not np.isfinite(X).all():
        raise ValueError("Expected a nonempty finite (n, 8) feature matrix")
    binary = np.asarray(gate.predict(X), dtype=int)
    if binary.shape != (len(X),) or not np.isin(binary, [0, 1]).all():
        raise ValueError("Binary LDA must output labels 0/1")
    final = np.zeros(len(X), dtype=int)
    routed = binary == 1
    if routed.any():
        channels = np.asarray(channel.predict(X[routed]), dtype=int)
        if not np.isin(channels, range(8)).all():
            raise ValueError("Channel LDA must output original labels 0..7")
        final[routed] = channels + 1
    return final, binary

import numpy as np


def series_errors(histories, targets, predictions, period):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 2 or len(histories) != len(targets):
        raise ValueError("M4 expects matching [series, horizon] arrays and one history per series")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Non-finite M4 targets or predictions")
    errors = np.abs(targets - predictions)
    denominators = np.abs(targets) + np.abs(predictions)
    if np.any(denominators == 0):
        raise ValueError("Undefined official sMAPE at target=prediction=0; inspect this series explicitly")
    smape = (200.0 * errors / denominators).mean(axis=1)
    scales = np.asarray([np.mean(np.abs(history[period:] - history[:-period])) if len(history) > period else np.nan for history in histories])
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Undefined official MASE scale; inspect constant or too-short training series")
    return smape, errors.mean(axis=1) / scales


def summarize_errors(smape, mase, naive2_smape, naive2_mase):
    arrays = [np.asarray(values, dtype=np.float64) for values in (smape, mase, naive2_smape, naive2_mase)]
    if not arrays[0].size or any(values.shape != arrays[0].shape or not np.isfinite(values).all() for values in arrays):
        raise ValueError("M4 aggregate requires nonempty, finite, matching per-series arrays")
    means = [float(values.mean()) for values in arrays]
    if means[2] <= 0 or means[3] <= 0:
        raise ValueError("Naive2 reference means must be positive")
    return {"sMAPE": means[0], "MASE": means[1], "OWA": 0.5 * (means[0] / means[2] + means[1] / means[3]), "Naive2_sMAPE": means[2], "Naive2_MASE": means[3], "series_count": len(arrays[0])}

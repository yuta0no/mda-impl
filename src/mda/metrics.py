import numpy as np
from jaxtyping import Float

Rates = Float[np.ndarray, " n"]

PASS_THRESHOLD = 0.05


def _sq_log_err(y_pred: Rates, y_true: Rates):
    return (np.log1p(y_pred) - np.log1p(y_true)) ** 2


def rmsle(y_pred: Rates, y_true: Rates) -> float:
    if not np.all(np.isfinite(y_pred)) or np.any(y_pred <= -1):
        return float("inf")
    return float(np.sqrt(np.mean(_sq_log_err(y_pred, y_true))))


def nmsle(y_pred: Rates, y_true: Rates) -> float:
    if not np.all(np.isfinite(y_pred)) or np.any(y_pred <= -1):
        return float("inf")
    return float(np.mean(_sq_log_err(y_pred, y_true)) / np.var(np.log1p(y_true)))


def passed(y_pred: Rates, y_true: Rates, threshold: float = PASS_THRESHOLD) -> bool:
    return rmsle(y_pred, y_true) < threshold

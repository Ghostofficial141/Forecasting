"""
metrics.py
==========
Centralised evaluation metrics for time-series forecasting.
All metric functions accept plain Python lists or numpy arrays.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = _validate(y_true, y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-9) -> float:
    """
    Mean Absolute Percentage Error.
    Adds epsilon to denominator to guard against division by zero.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-9) -> float:
    """Symmetric Mean Absolute Percentage Error (bounded between 0 and 200%)."""
    y_true, y_pred = _validate(y_true, y_pred)
    return float(
        np.mean(
            200 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon)
        )
    )


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    y_true, y_pred = _validate(y_true, y_pred)
    return float(r2_score(y_true, y_pred))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "unknown",
    state: str = "all",
) -> Dict[str, float]:
    """
    Compute all metrics and return as a dictionary.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Model predictions.
    model_name : str
        Label for the model.
    state : str
        State/group label.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: model, state, RMSE, MAE, MAPE, SMAPE, R2.
    """
    metrics = {
        "model": model_name,
        "state": state,
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    logger.info(
        f"[{model_name}][{state}] "
        f"RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f} | "
        f"MAPE={metrics['MAPE']:.2f}% | R2={metrics['R2']:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(y_true, y_pred):
    """Convert inputs to numpy float arrays and assert equal length."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    return y_true, y_pred

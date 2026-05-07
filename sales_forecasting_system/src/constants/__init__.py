"""
constants/__init__.py
=====================
Project-wide constants that are true constants (not config-driven).
"""

import os
from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]        # project root
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FORECASTS_DIR = ARTIFACTS_DIR / "forecasts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
LOGS_DIR = ROOT_DIR / "logs"
CONFIG_PATH = SRC_DIR / "config" / "config.yaml"

# ── Model names (canonical strings used across pipeline) ────────────────────
MODEL_ARIMA = "SARIMA"
MODEL_PROPHET = "Prophet"
MODEL_XGBOOST = "XGBoost"
MODEL_LSTM = "LSTM"
ALL_MODELS = [MODEL_ARIMA, MODEL_PROPHET, MODEL_XGBOOST, MODEL_LSTM]

# ── File extensions ──────────────────────────────────────────────────────────
PKL_EXT = ".pkl"
JSON_EXT = ".json"
CSV_EXT = ".csv"
PARQUET_EXT = ".parquet"

# ── Metric names ─────────────────────────────────────────────────────────────
METRIC_RMSE = "RMSE"
METRIC_MAE = "MAE"
METRIC_MAPE = "MAPE"
METRIC_SMAPE = "SMAPE"
METRIC_R2 = "R2"

# ── Ensure critical directories exist ────────────────────────────────────────
for _d in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    MODELS_DIR, METRICS_DIR, FORECASTS_DIR, PLOTS_DIR, LOGS_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

"""
model_training.py
=================
Orchestrates the training of all four models for each state:
  1. SARIMA  — univariate, per-state time series
  2. Prophet — univariate, per-state with seasonality / holidays
  3. XGBoost — feature-based, per-state with recursive forecasting
  4. LSTM    — deep learning, per-state with sequence modelling

Training is done with time-series-safe validation split (no shuffle).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.arima_model import SARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.utils.exception import ModelTrainingError
from src.utils.helpers import load_yaml, split_time_series
from src.utils.logger import get_logger
from src.constants import CONFIG_PATH

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains all configured models per state using a time-series split.
    Returns training/validation split info for use by ModelEvaluator.
    """

    def __init__(
        self,
        date_col: str,
        target_col: str,
        feature_cols: List[str],
        state_col: Optional[str] = None,
        config_path=None,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.state_col = state_col
        self.config = load_yaml(config_path or CONFIG_PATH)
        self.models_cfg = self.config["models"]
        self.test_ratio = self.config["data"]["test_ratio"]

        # Instantiate models
        self.sarima = SARIMAModel(config_path)
        self.prophet = ProphetModel(config_path)
        self.xgb = XGBoostModel(config_path)
        self.lstm = LSTMModel(config_path)

        # Store train/val splits per state for evaluation
        self.splits: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Train all enabled models on all states.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered, cleaned dataframe.

        Returns
        -------
        Dict[str, Dict]
            splits[state] = {"train": df_train, "val": df_val}
        """
        logger.info("=== MODEL TRAINING STARTED ===")

        states = (
            df[self.state_col].unique().tolist()
            if self.state_col and self.state_col in df.columns
            else ["__all__"]
        )
        logger.info(f"Training for {len(states)} state(s): {states}")

        for state in states:
            logger.info(f"── Training state: '{state}' ──")

            if self.state_col and self.state_col in df.columns:
                state_df = df[df[self.state_col] == state].copy()
            else:
                state_df = df.copy()

            state_df = state_df.sort_values(self.date_col).reset_index(drop=True)

            # Time-series-safe train/val split
            train_df, val_df = split_time_series(
                state_df, self.date_col, test_ratio=self.test_ratio
            )
            self.splits[state] = {"train": train_df, "val": val_df}

            logger.info(
                f"  Train: {len(train_df)} rows "
                f"({train_df[self.date_col].min()} → {train_df[self.date_col].max()})"
            )
            logger.info(
                f"  Val  : {len(val_df)} rows "
                f"({val_df[self.date_col].min()} → {val_df[self.date_col].max()})"
            )

            # ── SARIMA ──────────────────────────────────────────────────
            if self.models_cfg["arima"]["enabled"]:
                self._train_sarima(train_df, state)

            # ── Prophet ─────────────────────────────────────────────────
            if self.models_cfg["prophet"]["enabled"]:
                self._train_prophet(train_df, state)

            # ── XGBoost ─────────────────────────────────────────────────
            if self.models_cfg["xgboost"]["enabled"]:
                self._train_xgboost(train_df, state)

            # ── LSTM ────────────────────────────────────────────────────
            if self.models_cfg["lstm"]["enabled"]:
                self._train_lstm(train_df, state)

        logger.info("=== MODEL TRAINING COMPLETE ===")
        return self.splits

    # ------------------------------------------------------------------
    # Per-model training wrappers
    # ------------------------------------------------------------------

    def _train_sarima(self, train_df: pd.DataFrame, state: str) -> None:
        try:
            series = train_df.set_index(self.date_col)[self.target_col]
            series.index = pd.DatetimeIndex(series.index)
            self.sarima.fit(series, state=state)
        except Exception as e:
            logger.error(f"[SARIMA] Skipped for state='{state}': {e}")

    def _train_prophet(self, train_df: pd.DataFrame, state: str) -> None:
        try:
            self.prophet.fit(
                train_df,
                date_col=self.date_col,
                target_col=self.target_col,
                state=state,
            )
        except Exception as e:
            logger.error(f"[Prophet] Skipped for state='{state}': {e}")

    def _train_xgboost(self, train_df: pd.DataFrame, state: str) -> None:
        try:
            available_features = [
                c for c in self.feature_cols if c in train_df.columns
            ]
            # Drop rows with NaN in features
            clean = train_df[available_features + [self.target_col]].dropna()
            self.xgb.fit(
                clean,
                feature_cols=available_features,
                target_col=self.target_col,
                state=state,
            )
        except Exception as e:
            logger.error(f"[XGBoost] Skipped for state='{state}': {e}")

    def _train_lstm(self, train_df: pd.DataFrame, state: str) -> None:
        try:
            series = train_df[self.target_col].values.astype(float)
            self.lstm.fit(series, state=state)
        except Exception as e:
            logger.error(f"[LSTM] Skipped for state='{state}': {e}")

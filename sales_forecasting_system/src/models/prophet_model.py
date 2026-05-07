"""
prophet_model.py
================
Facebook Prophet wrapper for state-level sales forecasting.
Handles:
  - Yearly + weekly seasonality
  - US holidays
  - Changepoint detection
  - Confidence intervals
  - Per-state model serialisation
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.exception import ModelTrainingError, PredictionError
from src.utils.helpers import load_yaml, save_pickle, load_pickle, ensure_dir
from src.utils.logger import get_logger
from src.constants import CONFIG_PATH, MODELS_DIR

logger = get_logger(__name__)


class ProphetModel:
    """
    Facebook Prophet wrapper.
    Expects a dataframe with columns: ds (datetime), y (float target).
    """

    MODEL_NAME = "Prophet"

    def __init__(self, config_path=None):
        cfg = load_yaml(config_path or CONFIG_PATH)
        self.model_cfg = cfg["models"]["prophet"]
        self.horizon = cfg["models"]["forecast_horizon"]
        self.freq = cfg["data"]["frequency"]
        self._models: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        state: str = "all",
    ) -> None:
        """
        Fit Prophet on the given dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain `date_col` and `target_col`.
        date_col : str
            Name of the datetime column.
        target_col : str
            Name of the target column.
        state : str
            Label for serialisation.
        """
        logger.info(f"[Prophet] Fitting for state='{state}' | len={len(df)}")

        try:
            from prophet import Prophet

            # Prophet requires ds/y columns
            prophet_df = df[[date_col, target_col]].rename(
                columns={date_col: "ds", target_col: "y"}
            )
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
            prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")
            prophet_df = prophet_df.dropna()

            m = Prophet(
                yearly_seasonality=self.model_cfg.get("yearly_seasonality", True),
                weekly_seasonality=self.model_cfg.get("weekly_seasonality", True),
                daily_seasonality=self.model_cfg.get("daily_seasonality", False),
                changepoint_prior_scale=self.model_cfg.get("changepoint_prior_scale", 0.05),
                seasonality_prior_scale=self.model_cfg.get("seasonality_prior_scale", 10.0),
                interval_width=self.model_cfg.get("interval_width", 0.95),
            )

            # Add country holidays
            country = self.model_cfg.get("add_country_holidays", "US")
            if country:
                m.add_country_holidays(country_name=country)

            m.fit(prophet_df)
            self._models[state] = m
            self._save(state)
            logger.info(f"[Prophet][{state}] Training complete.")

        except Exception as e:
            raise ModelTrainingError(
                f"[Prophet] Training failed for state='{state}'", error=e
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        state: str = "all",
        n_periods: Optional[int] = None,
        last_date: Optional[pd.Timestamp] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Forecast the next n_periods steps.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            (point forecast array, full Prophet forecast DataFrame with CI)
        """
        n = n_periods or self.horizon
        m = self._get_model(state)

        try:
            freq_map = {"W": "W", "D": "D", "MS": "MS", "M": "MS"}
            freq = freq_map.get(self.freq, "W")
            future = m.make_future_dataframe(periods=n, freq=freq)
            forecast = m.predict(future)
            tail = forecast.tail(n)
            preds = np.maximum(tail["yhat"].values, 0)
            return preds.astype(float), tail

        except Exception as e:
            raise PredictionError(
                f"[Prophet] Prediction failed for state='{state}'", error=e
            )

    def predict_in_sample(
        self,
        df: pd.DataFrame,
        date_col: str,
        state: str = "all",
    ) -> np.ndarray:
        """Return in-sample predictions for evaluation."""
        m = self._get_model(state)
        try:
            hist_df = pd.DataFrame({"ds": pd.to_datetime(df[date_col].values)})
            forecast = m.predict(hist_df)
            return np.maximum(forecast["yhat"].values, 0).astype(float)
        except Exception as e:
            raise PredictionError(
                f"[Prophet] In-sample prediction failed for state='{state}'", error=e
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _save(self, state: str) -> None:
        ensure_dir(MODELS_DIR / self.MODEL_NAME)
        path = MODELS_DIR / self.MODEL_NAME / f"{state}.pkl"
        save_pickle(self._models[state], path)
        logger.debug(f"[Prophet] Model saved: {path}")

    def load(self, state: str) -> None:
        path = MODELS_DIR / self.MODEL_NAME / f"{state}.pkl"
        self._models[state] = load_pickle(path)
        logger.debug(f"[Prophet] Model loaded: {path}")

    def _get_model(self, state: str):
        if state not in self._models:
            self.load(state)
        return self._models[state]

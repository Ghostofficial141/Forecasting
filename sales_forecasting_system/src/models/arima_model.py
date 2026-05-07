"""
arima_model.py
==============
SARIMA / auto-ARIMA implementation.
Uses pmdarima.auto_arima for automatic (p, d, q)(P, D, Q, s) selection
based on AIC/BIC, with fallback to manual orders from config.
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


class SARIMAModel:
    """
    Wrapper around pmdarima auto_arima / statsmodels SARIMAX.
    Trained and serialised per state.
    """

    MODEL_NAME = "SARIMA"

    def __init__(self, config_path=None):
        cfg = load_yaml(config_path or CONFIG_PATH)
        self.model_cfg = cfg["models"]["arima"]
        self.horizon = cfg["models"]["forecast_horizon"]
        self.freq = cfg["data"]["frequency"]
        self._models: Dict[str, object] = {}   # state → fitted model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, series: pd.Series, state: str = "all") -> None:
        """
        Fit SARIMA on a univariate time series.

        Parameters
        ----------
        series : pd.Series
            Indexed by DatetimeIndex, values are the target.
        state : str
            State label used for serialisation.
        """
        logger.info(f"[SARIMA] Fitting for state='{state}' | len={len(series)}")
        try:
            import pmdarima as pm

            if self.model_cfg.get("auto_order", True):
                model = pm.auto_arima(
                    series.values,
                    start_p=1, start_q=1,
                    max_p=self.model_cfg.get("max_p", 3),
                    max_q=self.model_cfg.get("max_q", 3),
                    d=None,                # auto-select d
                    seasonal=self.model_cfg.get("seasonal", True),
                    m=self.model_cfg.get("s", 52),
                    information_criterion=self.model_cfg.get("information_criterion", "aic"),
                    stepwise=self.model_cfg.get("stepwise", True),
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                )
                logger.info(
                    f"[SARIMA][{state}] Best order={model.order} "
                    f"seasonal_order={model.seasonal_order} "
                    f"AIC={model.aic():.2f}"
                )
            else:
                # Manual order from config
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                p, d, q = self.model_cfg["p"], self.model_cfg["d"], self.model_cfg["q"]
                P, D, Q, s = (
                    self.model_cfg["P"], self.model_cfg["D"],
                    self.model_cfg["Q"], self.model_cfg["s"],
                )
                sarimax = SARIMAX(
                    series.values,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                model = sarimax.fit(disp=False)

            self._models[state] = model
            self._save(state)
            logger.info(f"[SARIMA][{state}] Training complete.")

        except Exception as e:
            raise ModelTrainingError(f"[SARIMA] Training failed for state='{state}'", error=e)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, state: str = "all", n_periods: Optional[int] = None) -> np.ndarray:
        """
        Generate future forecasts.

        Parameters
        ----------
        state : str
            State for which to generate forecasts.
        n_periods : int, optional
            Number of periods to forecast (defaults to config horizon).

        Returns
        -------
        np.ndarray
            1-D array of predictions.
        """
        n = n_periods or self.horizon
        model = self._get_model(state)

        try:
            import pmdarima as pm
            if isinstance(model, pm.arima.ARIMA):
                preds, _ = model.predict(n_periods=n, return_conf_int=True)
            else:
                # statsmodels SARIMAX result
                forecast = model.get_forecast(steps=n)
                preds = forecast.predicted_mean

            preds = np.maximum(preds, 0)   # clip negative sales
            return preds.astype(float)

        except Exception as e:
            raise PredictionError(
                f"[SARIMA] Prediction failed for state='{state}'", error=e
            )

    def predict_in_sample(
        self,
        series: pd.Series,
        state: str = "all",
    ) -> np.ndarray:
        """Return in-sample (training) predictions for evaluation."""
        model = self._get_model(state)
        try:
            import pmdarima as pm
            if isinstance(model, pm.arima.ARIMA):
                return np.maximum(model.predict_in_sample(), 0).astype(float)
            else:
                return np.maximum(model.fittedvalues, 0).astype(float)
        except Exception as e:
            raise PredictionError(
                f"[SARIMA] In-sample prediction failed for state='{state}'", error=e
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _save(self, state: str) -> None:
        ensure_dir(MODELS_DIR / self.MODEL_NAME)
        path = MODELS_DIR / self.MODEL_NAME / f"{state}.pkl"
        save_pickle(self._models[state], path)
        logger.debug(f"[SARIMA] Model saved: {path}")

    def load(self, state: str) -> None:
        path = MODELS_DIR / self.MODEL_NAME / f"{state}.pkl"
        self._models[state] = load_pickle(path)
        logger.debug(f"[SARIMA] Model loaded: {path}")

    def _get_model(self, state: str):
        if state not in self._models:
            self.load(state)
        return self._models[state]

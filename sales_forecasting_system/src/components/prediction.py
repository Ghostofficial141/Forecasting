"""
prediction.py
=============
Unified prediction interface that:
  1. Loads the best model selection metadata
  2. Routes each state's forecast request to the correct model
  3. Returns standardised forecast dicts with dates + values
  4. Exports results to CSV and JSON
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.components.model_selection import ModelSelector
from src.models.arima_model import SARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.utils.exception import PredictionError
from src.utils.helpers import load_yaml, save_json, ensure_dir
from src.utils.logger import get_logger
from src.constants import CONFIG_PATH, FORECASTS_DIR, MODEL_ARIMA, MODEL_PROPHET, MODEL_XGBOOST, MODEL_LSTM

logger = get_logger(__name__)


class Predictor:
    """
    Loads trained models and generates state-level forecasts.
    Supports routing to the best model or a specific named model.
    """

    def __init__(
        self,
        date_col: str,
        target_col: str,
        feature_cols: List[str],
        state_col: Optional[str],
        historical_df: pd.DataFrame,
        config_path=None,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.state_col = state_col
        self.historical_df = historical_df
        self.config = load_yaml(config_path or CONFIG_PATH)
        self.horizon = self.config["models"]["forecast_horizon"]
        self.freq = self.config["data"]["frequency"]

        # Lazy-loaded models
        self._sarima: Optional[SARIMAModel] = None
        self._prophet: Optional[ProphetModel] = None
        self._xgb: Optional[XGBoostModel] = None
        self._lstm: Optional[LSTMModel] = None

        ensure_dir(FORECASTS_DIR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_state(
        self,
        state: str,
        n_weeks: Optional[int] = None,
        model_override: Optional[str] = None,
    ) -> Dict:
        """
        Generate forecast for a single state.

        Parameters
        ----------
        state : str
            State name as it appears in the dataset.
        n_weeks : int, optional
            Number of weeks to forecast (overrides config if given).
        model_override : str, optional
            Force a specific model instead of the best-selected one.

        Returns
        -------
        Dict
            {
                "state": str,
                "model_used": str,
                "forecast_dates": List[str],
                "forecast_values": List[float],
                "generated_at": str,
            }
        """
        n = n_weeks or self.horizon
        model_name = model_override or ModelSelector.get_best_model_for_state(state)
        logger.info(f"Predicting state='{state}' using model='{model_name}' for {n} weeks")

        # Get state historical data
        if self.state_col and self.state_col in self.historical_df.columns:
            state_df = (
                self.historical_df[self.historical_df[self.state_col] == state]
                .sort_values(self.date_col)
                .reset_index(drop=True)
            )
        else:
            state_df = self.historical_df.sort_values(self.date_col).reset_index(drop=True)

        if len(state_df) == 0:
            raise PredictionError(f"No historical data found for state='{state}'")

        # Generate forecast dates
        last_date = pd.to_datetime(state_df[self.date_col].iloc[-1])
        forecast_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(self.freq),
            periods=n,
            freq=self.freq,
        )

        # Route to correct model
        preds = self._route_prediction(model_name, state, state_df, n)

        result = {
            "state": state,
            "model_used": model_name,
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "forecast_values": [round(float(v), 2) for v in preds],
            "generated_at": datetime.now().isoformat(),
        }

        # Save forecast
        self._save_forecast(result, state, model_name)
        return result

    def predict_all_states(
        self,
        n_weeks: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate forecasts for ALL states in the dataset.

        Returns
        -------
        List[Dict]
            List of forecast dicts, one per state.
        """
        logger.info("=== GENERATING FORECASTS FOR ALL STATES ===")

        if self.state_col and self.state_col in self.historical_df.columns:
            states = self.historical_df[self.state_col].unique().tolist()
        else:
            states = ["__all__"]

        all_forecasts = []
        for state in states:
            try:
                forecast = self.predict_state(state, n_weeks=n_weeks)
                all_forecasts.append(forecast)
            except Exception as e:
                logger.error(f"Forecast failed for state='{state}': {e}")

        # Save combined output
        save_json(all_forecasts, FORECASTS_DIR / "all_states_forecast.json")
        self._save_forecast_csv(all_forecasts)
        logger.info(
            f"=== FORECASTING COMPLETE — {len(all_forecasts)} states processed ==="
        )
        return all_forecasts

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _route_prediction(
        self,
        model_name: str,
        state: str,
        state_df: pd.DataFrame,
        n: int,
    ) -> np.ndarray:
        """Route the prediction to the correct model instance."""
        mn = model_name.upper()

        if "SARIMA" in mn or "ARIMA" in mn:
            return self._predict_sarima(state, n)

        elif "PROPHET" in mn:
            return self._predict_prophet(state, state_df, n)

        elif "XGBOOST" in mn or "XGB" in mn:
            return self._predict_xgboost(state, state_df, n)

        elif "LSTM" in mn:
            return self._predict_lstm(state, state_df, n)

        else:
            raise PredictionError(f"Unknown model: '{model_name}'")

    # ------------------------------------------------------------------
    # Model-specific wrappers (lazy-load)
    # ------------------------------------------------------------------

    def _predict_sarima(self, state: str, n: int) -> np.ndarray:
        if self._sarima is None:
            self._sarima = SARIMAModel()
        return self._sarima.predict(state=state, n_periods=n)

    def _predict_prophet(
        self, state: str, state_df: pd.DataFrame, n: int
    ) -> np.ndarray:
        if self._prophet is None:
            self._prophet = ProphetModel()
        preds, _ = self._prophet.predict(state=state, n_periods=n)
        return preds

    def _predict_xgboost(
        self, state: str, state_df: pd.DataFrame, n: int
    ) -> np.ndarray:
        if self._xgb is None:
            self._xgb = XGBoostModel()
        return self._xgb.predict(
            df=state_df,
            date_col=self.date_col,
            target_col=self.target_col,
            state=state,
            n_periods=n,
        )

    def _predict_lstm(
        self, state: str, state_df: pd.DataFrame, n: int
    ) -> np.ndarray:
        if self._lstm is None:
            self._lstm = LSTMModel()
        series = state_df[self.target_col].values.astype(float)
        return self._lstm.predict(series, state=state, n_periods=n)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_forecast(self, result: Dict, state: str, model_name: str) -> None:
        safe_state = state.replace("/", "_").replace(" ", "_")
        path = FORECASTS_DIR / f"{safe_state}_{model_name}_forecast.json"
        save_json(result, path)
        logger.debug(f"Forecast saved: {path}")

    def _save_forecast_csv(self, all_forecasts: List[Dict]) -> None:
        rows = []
        for f in all_forecasts:
            for date, val in zip(f["forecast_dates"], f["forecast_values"]):
                rows.append({
                    "state": f["state"],
                    "model_used": f["model_used"],
                    "forecast_date": date,
                    "forecast_sales": val,
                    "generated_at": f["generated_at"],
                })
        df = pd.DataFrame(rows)
        csv_path = FORECASTS_DIR / "all_states_forecast.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Forecast CSV saved: {csv_path}")

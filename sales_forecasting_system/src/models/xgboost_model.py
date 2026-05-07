"""
xgboost_model.py
================
XGBoost regression model for multi-step time-series forecasting.
Uses recursive (one-step-ahead) strategy:
  - Train on lag + rolling + calendar features
  - At inference time, predict t+1, append to history, predict t+2, ...
Includes feature importance logging.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.exception import ModelTrainingError, PredictionError
from src.utils.helpers import load_yaml, save_pickle, load_pickle, ensure_dir
from src.utils.logger import get_logger
from src.constants import CONFIG_PATH, MODELS_DIR

logger = get_logger(__name__)


class XGBoostModel:
    """
    XGBoost model with recursive multi-step forecasting.
    Expects pre-engineered feature dataframe (from FeatureEngineer).
    """

    MODEL_NAME = "XGBoost"

    def __init__(self, config_path=None):
        cfg = load_yaml(config_path or CONFIG_PATH)
        self.model_cfg = cfg["models"]["xgboost"]
        self.horizon = cfg["models"]["forecast_horizon"]
        self.feat_cfg = cfg.get("features", {})
        self.lag_periods: List[int] = self.feat_cfg.get("lag_periods", [1, 7, 30])
        self.rolling_windows: List[int] = self.feat_cfg.get("rolling_windows", [7, 30])
        self._models: Dict[str, object] = {}
        self._feature_cols: Dict[str, List[str]] = {}   # state → feature list

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        state: str = "all",
    ) -> None:
        """
        Fit XGBoost on feature-engineered data.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered dataframe (NaN rows already dropped).
        feature_cols : List[str]
            Names of input feature columns.
        target_col : str
            Name of the target column.
        state : str
            State label for serialisation.
        """
        logger.info(
            f"[XGBoost] Fitting for state='{state}' | "
            f"rows={len(df)} | features={len(feature_cols)}"
        )
        try:
            import xgboost as xgb
            from sklearn.model_selection import TimeSeriesSplit

            X = df[feature_cols].values
            y = df[target_col].values

            # Time-series cross-validation (just for parameter validation)
            tscv = TimeSeriesSplit(n_splits=3)

            params = {
                "n_estimators": self.model_cfg.get("n_estimators", 500),
                "max_depth": self.model_cfg.get("max_depth", 6),
                "learning_rate": self.model_cfg.get("learning_rate", 0.05),
                "subsample": self.model_cfg.get("subsample", 0.8),
                "colsample_bytree": self.model_cfg.get("colsample_bytree", 0.8),
                "min_child_weight": self.model_cfg.get("min_child_weight", 5),
                "gamma": self.model_cfg.get("gamma", 0.1),
                "reg_alpha": self.model_cfg.get("reg_alpha", 0.1),
                "reg_lambda": self.model_cfg.get("reg_lambda", 1.0),
                "random_state": self.model_cfg.get("random_state", 42),
                "objective": "reg:squarederror",
                "tree_method": "hist",
            }

            # Split last fold for early stopping
            split_idx = int(len(X) * 0.8)
            X_tr, X_val = X[:split_idx], X[split_idx:]
            y_tr, y_val = y[:split_idx], y[split_idx:]

            model = xgb.XGBRegressor(
                **params,
                early_stopping_rounds=self.model_cfg.get("early_stopping_rounds", 50),
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            self._models[state] = model
            self._feature_cols[state] = feature_cols
            self._save(state)

            # Log top-5 feature importances
            importances = dict(zip(feature_cols, model.feature_importances_))
            top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
            logger.info(f"[XGBoost][{state}] Top-5 feature importances: {top5}")
            logger.info(f"[XGBoost][{state}] Training complete.")

        except Exception as e:
            raise ModelTrainingError(
                f"[XGBoost] Training failed for state='{state}'", error=e
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        state: str = "all",
        n_periods: Optional[int] = None,
    ) -> np.ndarray:
        """
        Recursive multi-step forecasting.

        Strategy:
          1. Start from the last known row of `df`.
          2. Predict step t+1.
          3. Append prediction, regenerate features, predict t+2 … etc.

        Parameters
        ----------
        df : pd.DataFrame
            Historical dataframe with feature columns available.
        date_col : str
            Date column name.
        target_col : str
            Target column name.
        state : str
            State label.
        n_periods : int
            Number of future steps to predict.
        """
        n = n_periods or self.horizon
        model = self._get_model(state)
        feature_cols = self._feature_cols.get(state, [])

        try:
            freq = self.freq
            history = df.copy().sort_values(date_col).reset_index(drop=True)
            preds = []

            for _ in range(n):
                last_row = history.iloc[-1]
                last_date = pd.to_datetime(last_row[date_col])
                next_date = last_date + pd.tseries.frequencies.to_offset(freq)

                # Build next feature row
                next_row = self._build_next_row(history, next_date, date_col, target_col)
                X_next = np.array([[next_row.get(f, 0.0) for f in feature_cols]])

                pred = float(max(0.0, model.predict(X_next)[0]))
                preds.append(pred)

                # Append to history for next iteration
                new_row = {c: np.nan for c in history.columns}
                new_row[date_col] = next_date
                new_row[target_col] = pred
                for k, v in next_row.items():
                    new_row[k] = v

                history = pd.concat(
                    [history, pd.DataFrame([new_row])],
                    ignore_index=True,
                )

            return np.array(preds, dtype=float)

        except Exception as e:
            raise PredictionError(
                f"[XGBoost] Prediction failed for state='{state}'", error=e
            )

    def predict_in_sample(
        self,
        df: pd.DataFrame,
        target_col: str,
        state: str = "all",
    ) -> np.ndarray:
        """Return in-sample predictions for evaluation."""
        model = self._get_model(state)
        feature_cols = self._feature_cols.get(state, [])
        X = df[feature_cols].values
        return np.maximum(model.predict(X), 0).astype(float)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _build_next_row(
        self,
        history: pd.DataFrame,
        next_date: pd.Timestamp,
        date_col: str,
        target_col: str,
    ) -> Dict:
        """
        Compute feature values for the next unseen timestep
        using the current history buffer.
        """
        row: Dict = {}
        target_series = history[target_col].values.astype(float)

        # Lag features
        for lag in self.lag_periods:
            row[f"lag_{lag}"] = target_series[-lag] if len(target_series) >= lag else np.nan

        # Rolling features
        for w in self.rolling_windows:
            window_data = target_series[-w:] if len(target_series) >= w else target_series
            row[f"rolling_mean_{w}"] = float(np.mean(window_data))
            row[f"rolling_std_{w}"] = float(np.std(window_data))

        # Calendar features
        row["day_of_week"] = next_date.dayofweek
        row["week_of_year"] = next_date.isocalendar()[1]
        row["month"] = next_date.month
        row["quarter"] = next_date.quarter
        row["year"] = next_date.year
        row["is_weekend"] = int(next_date.dayofweek >= 5)

        # Holiday flag (US focused)
        import holidays
        us_holidays = holidays.US()
        row["holiday_flag"] = 1 if next_date in us_holidays else 0

        return row

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _save(self, state: str) -> None:
        ensure_dir(MODELS_DIR / self.MODEL_NAME)
        save_pickle(self._models[state], MODELS_DIR / self.MODEL_NAME / f"{state}.pkl")
        save_pickle(self._feature_cols[state], MODELS_DIR / self.MODEL_NAME / f"{state}_features.pkl")

    def load(self, state: str) -> None:
        self._models[state] = load_pickle(MODELS_DIR / self.MODEL_NAME / f"{state}.pkl")
        feat_path = MODELS_DIR / self.MODEL_NAME / f"{state}_features.pkl"
        if feat_path.exists():
            self._feature_cols[state] = load_pickle(feat_path)

    def _get_model(self, state: str):
        if state not in self._models:
            self.load(state)
        return self._models[state]

"""
model_evaluation.py
===================
Evaluates all trained models on the validation split.
Generates:
  - Metrics DataFrame (RMSE, MAE, MAPE, SMAPE, R2 per model per state)
  - Prediction vs Actual plots
  - Saves metrics to artifacts/metrics/
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.models.arima_model import SARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.utils.metrics import compute_all_metrics
from src.utils.helpers import save_json, ensure_dir
from src.utils.logger import get_logger
from src.constants import METRICS_DIR, PLOTS_DIR, CONFIG_PATH
from src.utils.helpers import load_yaml

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Runs all models on the held-out validation set,
    computes standardised metrics, and produces visualisations.
    """

    def __init__(
        self,
        date_col: str,
        target_col: str,
        feature_cols: List[str],
        state_col: Optional[str],
        sarima: SARIMAModel,
        prophet: ProphetModel,
        xgb: XGBoostModel,
        lstm: LSTMModel,
        config_path=None,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.state_col = state_col
        self.sarima = sarima
        self.prophet = prophet
        self.xgb = xgb
        self.lstm = lstm
        self.config = load_yaml(config_path or CONFIG_PATH)
        self.models_cfg = self.config["models"]
        ensure_dir(METRICS_DIR)
        ensure_dir(PLOTS_DIR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, splits: Dict[str, Dict]) -> pd.DataFrame:
        """
        Evaluate all models on all states.

        Parameters
        ----------
        splits : Dict
            Output from ModelTrainer.run(): {state: {"train": ..., "val": ...}}

        Returns
        -------
        pd.DataFrame
            Metrics for every (model, state) combination.
        """
        logger.info("=== MODEL EVALUATION STARTED ===")
        all_metrics = []

        for state, split in splits.items():
            train_df = split["train"]
            val_df = split["val"]

            if len(val_df) == 0:
                logger.warning(f"Empty validation set for state='{state}'. Skipping.")
                continue

            y_true = val_df[self.target_col].values.astype(float)
            logger.info(f"── Evaluating state: '{state}' | val_len={len(val_df)} ──")

            # ── SARIMA ──────────────────────────────────────────────────
            if self.models_cfg["arima"]["enabled"]:
                metrics = self._eval_sarima(state, val_df, y_true, train_df)
                if metrics:
                    all_metrics.append(metrics)

            # ── Prophet ─────────────────────────────────────────────────
            if self.models_cfg["prophet"]["enabled"]:
                metrics = self._eval_prophet(state, train_df, val_df, y_true)
                if metrics:
                    all_metrics.append(metrics)

            # ── XGBoost ─────────────────────────────────────────────────
            if self.models_cfg["xgboost"]["enabled"]:
                metrics = self._eval_xgboost(state, val_df, y_true)
                if metrics:
                    all_metrics.append(metrics)

            # ── LSTM ────────────────────────────────────────────────────
            if self.models_cfg["lstm"]["enabled"]:
                metrics = self._eval_lstm(state, train_df, val_df, y_true)
                if metrics:
                    all_metrics.append(metrics)

        metrics_df = pd.DataFrame(all_metrics)
        if not metrics_df.empty:
            metrics_df = metrics_df.sort_values(["state", "RMSE"]).reset_index(drop=True)

        # Save metrics
        metrics_path = METRICS_DIR / "all_model_metrics.json"
        save_json(metrics_df.to_dict(orient="records"), metrics_path)
        logger.info(f"Metrics saved to: {metrics_path}")

        # Save CSV
        metrics_df.to_csv(METRICS_DIR / "all_model_metrics.csv", index=False)

        logger.info("=== MODEL EVALUATION COMPLETE ===")
        logger.info(f"\n{metrics_df.to_string(index=False)}")
        return metrics_df

    # ------------------------------------------------------------------
    # Per-model evaluation helpers
    # ------------------------------------------------------------------

    def _eval_sarima(self, state, val_df, y_true, train_df) -> Optional[Dict]:
        try:
            n = len(val_df)
            preds = self.sarima.predict(state=state, n_periods=n)
            preds = preds[:n]
            m = compute_all_metrics(y_true, preds, model_name="SARIMA", state=state)
            self._plot_prediction(
                val_df[self.date_col].values, y_true, preds, state, "SARIMA"
            )
            return m
        except Exception as e:
            logger.error(f"[SARIMA] Evaluation failed for state='{state}': {e}")
            return None

    def _eval_prophet(self, state, train_df, val_df, y_true) -> Optional[Dict]:
        try:
            preds = self.prophet.predict_in_sample(val_df, self.date_col, state=state)
            preds = preds[:len(y_true)]
            m = compute_all_metrics(y_true, preds, model_name="Prophet", state=state)
            self._plot_prediction(
                val_df[self.date_col].values, y_true, preds, state, "Prophet"
            )
            return m
        except Exception as e:
            logger.error(f"[Prophet] Evaluation failed for state='{state}': {e}")
            return None

    def _eval_xgboost(self, state, val_df, y_true) -> Optional[Dict]:
        try:
            available = [c for c in self.feature_cols if c in val_df.columns]
            clean_val = val_df[available + [self.target_col]].dropna()
            if len(clean_val) == 0:
                return None
            preds = self.xgb.predict_in_sample(clean_val, self.target_col, state=state)
            y_t = clean_val[self.target_col].values.astype(float)
            m = compute_all_metrics(y_t, preds, model_name="XGBoost", state=state)
            self._plot_prediction(
                clean_val.index.values, y_t, preds, state, "XGBoost"
            )
            return m
        except Exception as e:
            logger.error(f"[XGBoost] Evaluation failed for state='{state}': {e}")
            return None

    def _eval_lstm(self, state, train_df, val_df, y_true) -> Optional[Dict]:
        try:
            # Predict val_len steps from end of training series
            series = train_df[self.target_col].values.astype(float)
            n = len(val_df)
            preds = self.lstm.predict(series, state=state, n_periods=n)
            preds = preds[:n]
            m = compute_all_metrics(y_true, preds, model_name="LSTM", state=state)
            self._plot_prediction(
                val_df[self.date_col].values, y_true, preds, state, "LSTM"
            )
            return m
        except Exception as e:
            logger.error(f"[LSTM] Evaluation failed for state='{state}': {e}")
            return None

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_prediction(
        self,
        dates: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        state: str,
        model_name: str,
    ) -> None:
        """Save Actual vs Predicted plot for a given model and state."""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(
                f"{model_name} — State: {state}\nActual vs Predicted",
                fontsize=14, fontweight="bold",
            )

            # ── Top: Line plot ────────────────────────────────────────
            ax = axes[0]
            ax.plot(dates, y_true, label="Actual", color="#2196F3", linewidth=2)
            ax.plot(dates, y_pred, label="Predicted", color="#FF5722",
                    linewidth=2, linestyle="--")
            ax.fill_between(
                dates,
                np.minimum(y_true, y_pred),
                np.maximum(y_true, y_pred),
                alpha=0.15, color="#FF9800",
            )
            ax.set_ylabel("Sales")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # ── Bottom: Residual plot ─────────────────────────────────
            ax2 = axes[1]
            residuals = y_true - y_pred
            ax2.bar(range(len(residuals)), residuals, color="#9C27B0", alpha=0.7)
            ax2.axhline(0, color="black", linewidth=1)
            ax2.set_xlabel("Time step (validation)")
            ax2.set_ylabel("Residual")
            ax2.set_title("Prediction Errors")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            safe_state = state.replace("/", "_").replace(" ", "_")
            save_path = PLOTS_DIR / f"{model_name}_{safe_state}_eval.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.debug(f"Plot saved: {save_path}")

        except Exception as e:
            logger.warning(f"Failed to save plot for {model_name}/{state}: {e}")

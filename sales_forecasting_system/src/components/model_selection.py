"""
model_selection.py
==================
Selects the best model per state based on the primary evaluation metric
(default: RMSE, lower-is-better).
Saves the selection metadata as JSON so the API can load it at startup.
"""

from typing import Dict, Optional

import pandas as pd

from src.utils.exception import ModelSelectionError
from src.utils.helpers import save_json, load_json, load_yaml, ensure_dir
from src.utils.logger import get_logger
from src.constants import METRICS_DIR, CONFIG_PATH

logger = get_logger(__name__)

SELECTION_FILE = METRICS_DIR / "best_model_selection.json"


class ModelSelector:
    """
    Picks the best model per state from a metrics DataFrame.
    """

    def __init__(self, config_path=None):
        cfg = load_yaml(config_path or CONFIG_PATH)
        sel_cfg = cfg.get("selection", {})
        self.primary_metric: str = sel_cfg.get("primary_metric", "RMSE")
        self.lower_is_better: bool = sel_cfg.get("lower_is_better", True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, metrics_df: pd.DataFrame) -> Dict[str, str]:
        """
        Select best model per state.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Output from ModelEvaluator — columns: model, state, RMSE, MAE, ...

        Returns
        -------
        Dict[str, str]
            {state: best_model_name}
        """
        logger.info("=== MODEL SELECTION STARTED ===")

        if metrics_df.empty:
            raise ModelSelectionError("Metrics DataFrame is empty. Cannot select best model.")

        if self.primary_metric not in metrics_df.columns:
            raise ModelSelectionError(
                f"Primary metric '{self.primary_metric}' not found in metrics. "
                f"Available: {metrics_df.columns.tolist()}"
            )

        selection: Dict[str, str] = {}

        for state, grp in metrics_df.groupby("state"):
            grp = grp.dropna(subset=[self.primary_metric])
            if grp.empty:
                logger.warning(f"No valid metrics for state='{state}'. Skipping.")
                continue

            if self.lower_is_better:
                best_row = grp.loc[grp[self.primary_metric].idxmin()]
            else:
                best_row = grp.loc[grp[self.primary_metric].idxmax()]

            best_model = best_row["model"]
            best_score = best_row[self.primary_metric]
            selection[str(state)] = str(best_model)

            logger.info(
                f"  [{state}] Best model: {best_model} "
                f"({self.primary_metric}={best_score:.4f})"
            )

        # Save selection metadata
        ensure_dir(METRICS_DIR)
        save_json(selection, SELECTION_FILE)
        logger.info(f"Best model selection saved to: {SELECTION_FILE}")

        # Also save a rich summary
        summary = []
        for state, model in selection.items():
            row = metrics_df[
                (metrics_df["state"] == state) & (metrics_df["model"] == model)
            ]
            if not row.empty:
                summary.append(row.iloc[0].to_dict())

        save_json(summary, METRICS_DIR / "best_model_metrics.json")
        logger.info("=== MODEL SELECTION COMPLETE ===")
        return selection

    # ------------------------------------------------------------------
    # Load saved selection
    # ------------------------------------------------------------------

    @staticmethod
    def load_selection() -> Dict[str, str]:
        """Load the saved model selection from disk."""
        if not SELECTION_FILE.exists():
            raise ModelSelectionError(
                f"Model selection file not found: {SELECTION_FILE}. "
                "Please run the training pipeline first."
            )
        return load_json(SELECTION_FILE)

    @staticmethod
    def get_best_model_for_state(state: str) -> str:
        """Return the best model name for a given state."""
        selection = ModelSelector.load_selection()
        # Try exact match first, then case-insensitive
        if state in selection:
            return selection[state]
        lower_map = {k.lower(): v for k, v in selection.items()}
        if state.lower() in lower_map:
            return lower_map[state.lower()]
        # Fallback: return the globally most common best model
        if selection:
            from collections import Counter
            most_common = Counter(selection.values()).most_common(1)[0][0]
            logger.warning(
                f"State '{state}' not found in selection. "
                f"Falling back to globally best model: {most_common}"
            )
            return most_common
        raise ModelSelectionError(
            f"State '{state}' not found and no fallback available."
        )

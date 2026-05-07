"""
prediction_pipeline.py
======================
Inference-only pipeline:
  Loads processed data → loads best model selection → generates forecasts
  → exports CSV/JSON

Can be called from CLI or API independently of training.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.components.feature_engineering import FeatureEngineer
from src.components.model_selection import ModelSelector
from src.components.prediction import Predictor
from src.utils.helpers import load_yaml, load_json, ensure_dir
from src.utils.logger import get_logger
from src.constants import CONFIG_PATH, FORECASTS_DIR

logger = get_logger(__name__)


class PredictionPipeline:
    """
    Loads all necessary artifacts and runs inference for all states
    or a specific state+weeks request.
    """

    def __init__(self, config_path=None):
        self.config_path = config_path or CONFIG_PATH
        self.config = load_yaml(self.config_path)
        self._historical_df: Optional[pd.DataFrame] = None
        self._column_map: Optional[Dict] = None
        self._feature_cols: Optional[List[str]] = None
        self._predictor: Optional[Predictor] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load processed / feature-engineered data from disk."""
        if self._historical_df is not None:
            return   # already loaded

        processed_path = Path(self.config["data"]["processed_data_path"])
        if not processed_path.is_absolute():
            processed_path = Path(__file__).resolve().parents[2] / processed_path

        if not processed_path.exists():
            raise FileNotFoundError(
                f"Processed data not found at {processed_path}. "
                "Run the training pipeline first: python main.py train"
            )

        self._historical_df = pd.read_parquet(processed_path)

        # Resolve column names from config (or re-detect)
        data_cfg = self.config["data"]
        from src.utils.helpers import (
            detect_date_column, detect_target_column, detect_state_column
        )
        date_col = data_cfg.get("date_column") or detect_date_column(self._historical_df)
        target_col = data_cfg.get("target_column") or detect_target_column(self._historical_df)
        state_col = data_cfg.get("state_column") or detect_state_column(self._historical_df)

        self._column_map = {
            "date": date_col,
            "target": target_col,
            "state": state_col,
        }

        # Re-apply feature engineering (without drop_na so history is complete)
        feat_eng = FeatureEngineer(date_col, target_col, state_col, self.config_path)
        self._historical_df = feat_eng.run(self._historical_df, drop_na=False)
        self._feature_cols = feat_eng.get_feature_columns(self._historical_df)

        logger.info(
            f"Loaded processed data: shape={self._historical_df.shape} | "
            f"features={len(self._feature_cols)}"
        )

    def _get_predictor(self) -> Predictor:
        """Return (cached) Predictor instance."""
        if self._predictor is None:
            self._load_data()
            self._predictor = Predictor(
                date_col=self._column_map["date"],
                target_col=self._column_map["target"],
                feature_cols=self._feature_cols,
                state_col=self._column_map["state"],
                historical_df=self._historical_df,
                config_path=self.config_path,
            )
        return self._predictor

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
        Called by the /predict API endpoint.
        """
        predictor = self._get_predictor()
        return predictor.predict_state(state, n_weeks=n_weeks, model_override=model_override)

    def predict_all(self, n_weeks: Optional[int] = None) -> List[Dict]:
        """Generate forecasts for all states."""
        predictor = self._get_predictor()
        return predictor.predict_all_states(n_weeks=n_weeks)

    def run(self) -> List[Dict]:
        """CLI entry point: forecast all states and return results."""
        logger.info("=== PREDICTION PIPELINE STARTED ===")
        results = self.predict_all()
        logger.info(f"=== PREDICTION PIPELINE COMPLETE — {len(results)} forecasts ===")
        return results

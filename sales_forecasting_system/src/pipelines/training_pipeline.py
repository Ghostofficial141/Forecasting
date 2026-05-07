"""
training_pipeline.py
====================
End-to-end training pipeline:
  DataIngestion → DataValidation → Preprocessing → FeatureEngineering
  → ModelTraining → ModelEvaluation → ModelSelection

Designed to be run from CLI (main.py) or called from the API /train endpoint.
Returns a summary dict that the API can serialise to JSON.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidator
from src.components.preprocessing import DataPreprocessor
from src.components.feature_engineering import FeatureEngineer
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluator
from src.components.model_selection import ModelSelector
from src.utils.helpers import load_yaml, save_json, get_timestamp_str, ensure_dir
from src.utils.logger import get_logger
from src.constants import CONFIG_PATH, ARTIFACTS_DIR

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Orchestrates the full training workflow in a single .run() call.
    All intermediate artifacts are saved to disk for reproducibility.
    """

    def __init__(self, config_path=None):
        self.config_path = config_path or CONFIG_PATH
        self.config = load_yaml(self.config_path)
        ensure_dir(ARTIFACTS_DIR)

    def run(self) -> Dict:
        """
        Execute the complete training pipeline.

        Returns
        -------
        Dict
            Summary: best models per state, metrics path, run timestamp.
        """
        ts = get_timestamp_str()
        logger.info(f"{'='*60}")
        logger.info(f"  TRAINING PIPELINE STARTED — {ts}")
        logger.info(f"{'='*60}")

        # ── Step 1: Data Ingestion ───────────────────────────────────
        logger.info("STEP 1/7 — Data Ingestion")
        ingestion = DataIngestion(self.config_path)
        raw_df, column_map = ingestion.run()
        date_col = column_map["date"]
        target_col = column_map["target"]
        state_col = column_map.get("state")

        # ── Step 2: Data Validation ──────────────────────────────────
        logger.info("STEP 2/7 — Data Validation")
        validator = DataValidator(date_col, target_col, state_col)
        validation_report = validator.run(raw_df)

        # ── Step 3: Preprocessing ────────────────────────────────────
        logger.info("STEP 3/7 — Preprocessing")
        preprocessor = DataPreprocessor(date_col, target_col, state_col, self.config_path)
        clean_df = preprocessor.run(raw_df)
        preprocessor.fit_scalers(clean_df)

        # ── Step 4: Feature Engineering ──────────────────────────────
        logger.info("STEP 4/7 — Feature Engineering")
        feat_eng = FeatureEngineer(date_col, target_col, state_col, self.config_path)
        featured_df = feat_eng.run(clean_df, drop_na=True)
        feature_cols = feat_eng.get_feature_columns(featured_df)

        # Save processed data for API use
        processed_path = Path(self.config["data"]["processed_data_path"])
        if not processed_path.is_absolute():
            processed_path = Path(__file__).resolve().parents[2] / processed_path
        ensure_dir(processed_path.parent)
        featured_df.to_parquet(processed_path, index=False)
        logger.info(f"Feature-engineered data saved: {processed_path}")

        # ── Step 5: Model Training ───────────────────────────────────
        logger.info("STEP 5/7 — Model Training")
        trainer = ModelTrainer(
            date_col, target_col, feature_cols, state_col, self.config_path
        )
        splits = trainer.run(featured_df)

        # ── Step 6: Model Evaluation ─────────────────────────────────
        logger.info("STEP 6/7 — Model Evaluation")
        evaluator = ModelEvaluator(
            date_col, target_col, feature_cols, state_col,
            sarima=trainer.sarima,
            prophet=trainer.prophet,
            xgb=trainer.xgb,
            lstm=trainer.lstm,
            config_path=self.config_path,
        )
        metrics_df = evaluator.run(splits)

        # ── Step 7: Model Selection ──────────────────────────────────
        logger.info("STEP 7/7 — Model Selection")
        selector = ModelSelector(self.config_path)
        best_models = selector.run(metrics_df)

        # ── Build summary ────────────────────────────────────────────
        summary = {
            "pipeline_run": ts,
            "status": "success",
            "column_map": column_map,
            "feature_columns": feature_cols,
            "states_trained": list(best_models.keys()),
            "best_models": best_models,
            "total_metrics_computed": len(metrics_df),
            "validation_report": {
                k: v for k, v in validation_report.items() if k != "stats"
            },
        }

        save_json(summary, ARTIFACTS_DIR / f"pipeline_run_{ts}.json")

        logger.info(f"{'='*60}")
        logger.info(f"  TRAINING PIPELINE COMPLETE — {ts}")
        logger.info(f"  Best models: {best_models}")
        logger.info(f"{'='*60}")

        return summary

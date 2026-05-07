"""
main.py
=======
CLI entry point for the Sales Forecasting System.

Usage:
  python main.py train        — Run full training pipeline
  python main.py predict      — Generate forecasts for all states
  python main.py api          — Start the REST API server
  python main.py all          — Train + predict
"""

import argparse
import sys
import os

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import get_logger
from src.constants import CONFIG_PATH

logger = get_logger("main")


def run_training():
    """Execute the full end-to-end training pipeline."""
    logger.info("Starting training pipeline via CLI…")
    from src.pipelines.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline(CONFIG_PATH)
    summary = pipeline.run()
    logger.info(f"Training complete. Best models: {summary['best_models']}")
    return summary


def run_prediction():
    """Generate forecasts for all states."""
    logger.info("Starting prediction pipeline via CLI…")
    from src.pipelines.prediction_pipeline import PredictionPipeline
    pipeline = PredictionPipeline(CONFIG_PATH)
    results = pipeline.run()
    logger.info(f"Prediction complete. Forecasts generated for {len(results)} states.")
    return results


def run_api():
    """Start the FastAPI server with uvicorn."""
    from src.utils.helpers import load_yaml
    cfg = load_yaml(CONFIG_PATH)
    api_cfg = cfg.get("api", {})

    import uvicorn
    logger.info(
        f"Starting API at http://{api_cfg.get('host', '0.0.0.0')}:"
        f"{api_cfg.get('port', 8000)}"
    )
    uvicorn.run(
        "api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", False),
        log_level=api_cfg.get("log_level", "info"),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Sales Forecasting System — CLI",
    )
    parser.add_argument(
        "command",
        choices=["train", "predict", "api", "all"],
        help=(
            "train: Run training pipeline | "
            "predict: Generate forecasts | "
            "api: Start REST API | "
            "all: Train then predict"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "train":
        run_training()

    elif args.command == "predict":
        run_prediction()

    elif args.command == "api":
        run_api()

    elif args.command == "all":
        run_training()
        run_prediction()

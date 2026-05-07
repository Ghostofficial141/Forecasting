"""
routes/forecast.py
==================
FastAPI router for all forecasting-related endpoints:
  POST /predict
  POST /predict/all
  POST /train
  GET  /metrics
  GET  /metrics/best
  GET  /models
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from api.schemas.request import PredictRequest, TrainRequest
from api.schemas.response import (
    AllForecastsResponse,
    BestModelResponse,
    ErrorResponse,
    ForecastPoint,
    MetricsResponse,
    ModelInfo,
    ModelMetric,
    ModelsListResponse,
    PredictResponse,
    TrainResponse,
)
from src.components.model_selection import ModelSelector
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.helpers import load_json
from src.utils.logger import get_logger
from src.constants import METRICS_DIR, ALL_MODELS

logger = get_logger(__name__)
router = APIRouter(prefix="", tags=["Forecasting"])

# Thread pool for running blocking ML code in async context
_executor = ThreadPoolExecutor(max_workers=4)

# Shared prediction pipeline (initialised lazily per request)
_pred_pipeline: Optional[PredictionPipeline] = None


def _get_prediction_pipeline() -> PredictionPipeline:
    global _pred_pipeline
    if _pred_pipeline is None:
        _pred_pipeline = PredictionPipeline()
    return _pred_pipeline


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Generate sales forecast for a state",
    description=(
        "Returns the next N weeks of sales forecast for the given US state. "
        "Uses the best trained model by default. "
        "Optionally override with a specific model."
    ),
    responses={
        200: {"description": "Forecast generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        404: {"model": ErrorResponse, "description": "State not found in dataset"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def predict(body: PredictRequest):
    """
    **POST /predict**

    Request body:
    ```json
    {
        "state": "California",
        "weeks": 8,
        "model": null
    }
    ```
    """
    logger.info(f"POST /predict — state='{body.state}', weeks={body.weeks}")
    pipeline = _get_prediction_pipeline()

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor,
            lambda: pipeline.predict_state(
                state=body.state,
                n_weeks=body.weeks,
                model_override=body.model,
            ),
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Models not trained yet. Run POST /train first. ({e})",
        )
    except Exception as e:
        logger.error(f"Prediction error for state='{body.state}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    # Build response
    forecast_points = [
        ForecastPoint(date=d, sales=v)
        for d, v in zip(result["forecast_dates"], result["forecast_values"])
    ]
    return PredictResponse(
        state=result["state"],
        model_used=result["model_used"],
        weeks=len(forecast_points),
        forecast=forecast_points,
        generated_at=result["generated_at"],
    )


# ---------------------------------------------------------------------------
# POST /predict/all
# ---------------------------------------------------------------------------

@router.post(
    "/predict/all",
    response_model=AllForecastsResponse,
    summary="Generate forecasts for ALL states",
)
async def predict_all(weeks: int = Query(default=8, ge=1, le=52)):
    """Generate forecast for every state in the dataset."""
    logger.info(f"POST /predict/all — weeks={weeks}")
    pipeline = _get_prediction_pipeline()
    loop = asyncio.get_event_loop()

    try:
        results = await loop.run_in_executor(
            _executor,
            lambda: pipeline.predict_all(n_weeks=weeks),
        )
    except Exception as e:
        logger.error(f"Error in predict_all: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    forecasts = []
    for r in results:
        points = [
            ForecastPoint(date=d, sales=v)
            for d, v in zip(r["forecast_dates"], r["forecast_values"])
        ]
        forecasts.append(
            PredictResponse(
                state=r["state"],
                model_used=r["model_used"],
                weeks=len(points),
                forecast=points,
                generated_at=r["generated_at"],
            )
        )

    return AllForecastsResponse(forecasts=forecasts, total_states=len(forecasts))


# ---------------------------------------------------------------------------
# POST /train
# ---------------------------------------------------------------------------

@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Retrain all models",
    description=(
        "Triggers the full end-to-end training pipeline: "
        "ingestion → validation → preprocessing → feature engineering → "
        "training → evaluation → model selection."
    ),
)
async def train(body: TrainRequest = TrainRequest()):
    """Retrain all models on the latest data."""
    global _pred_pipeline
    logger.info(f"POST /train — force_retrain={body.force_retrain}")

    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(
            _executor,
            lambda: TrainingPipeline().run(),
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Reset the prediction pipeline so it loads fresh artifacts
    _pred_pipeline = None

    return TrainResponse(
        pipeline_run=summary["pipeline_run"],
        states_trained=summary["states_trained"],
        best_models=summary["best_models"],
        total_metrics_computed=summary["total_metrics_computed"],
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Return all model evaluation metrics",
)
async def get_metrics():
    """Return RMSE, MAE, MAPE, SMAPE, R2 for all trained models."""
    metrics_file = METRICS_DIR / "all_model_metrics.json"
    if not metrics_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run POST /train first.",
        )
    try:
        data = load_json(metrics_file)
        metrics = [ModelMetric(**row) for row in data]
        return MetricsResponse(metrics=metrics, total_entries=len(metrics))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /metrics/best
# ---------------------------------------------------------------------------

@router.get(
    "/metrics/best",
    response_model=BestModelResponse,
    summary="Return the best model per state",
)
async def get_best_models():
    """Return the model selection: {state → best_model_name}."""
    try:
        selection = ModelSelector.load_selection()
        return BestModelResponse(best_models=selection)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model selection not found. Run POST /train first. ({e})",
        )


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List all available forecasting models",
)
async def list_models():
    """Return metadata about every supported model."""
    model_info = [
        ModelInfo(
            name="SARIMA",
            description="Seasonal ARIMA with automatic order selection via pmdarima.",
            type="Statistical",
        ),
        ModelInfo(
            name="Prophet",
            description="Facebook Prophet with yearly/weekly seasonality and US holidays.",
            type="Statistical / ML",
        ),
        ModelInfo(
            name="XGBoost",
            description="Gradient-boosted trees using lag + rolling + calendar features; recursive forecasting.",
            type="Machine Learning",
        ),
        ModelInfo(
            name="LSTM",
            description="Multi-layer LSTM with dropout; sequence-to-one; recursive multi-step.",
            type="Deep Learning",
        ),
    ]
    return ModelsListResponse(models=model_info)

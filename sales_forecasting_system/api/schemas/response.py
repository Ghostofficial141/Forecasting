"""
schemas/response.py
===================
Pydantic v2 response schemas for the forecasting REST API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class BaseResponse(BaseModel):
    """Common envelope for all API responses."""
    status: str = Field(default="success", description="Response status: 'success' or 'error'.")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="UTC timestamp of the response.",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseResponse):
    service: str = "Sales Forecasting API"
    version: str = "1.0.0"
    models_ready: bool = Field(
        description="True if trained model artifacts are present."
    )
    message: str = ""


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class ForecastPoint(BaseModel):
    """Single forecast data point."""
    date: str = Field(description="Forecast date in YYYY-MM-DD format.")
    sales: float = Field(description="Predicted weekly sales value.")


class PredictResponse(BaseResponse):
    state: str = Field(description="State for which the forecast was generated.")
    model_used: str = Field(description="Name of the model used for forecasting.")
    weeks: int = Field(description="Number of forecast weeks.")
    forecast: List[ForecastPoint] = Field(description="List of forecast points.")
    generated_at: str = Field(description="ISO timestamp when forecast was generated.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class ModelMetric(BaseModel):
    """Metrics for a single model/state combination."""
    model: str
    state: str
    RMSE: Optional[float] = None
    MAE: Optional[float] = None
    MAPE: Optional[float] = None
    SMAPE: Optional[float] = None
    R2: Optional[float] = None


class MetricsResponse(BaseResponse):
    metrics: List[ModelMetric]
    total_entries: int


class BestModelResponse(BaseResponse):
    best_models: Dict[str, str] = Field(
        description="Mapping of state → best model name."
    )


# ---------------------------------------------------------------------------
# Available models
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    name: str
    description: str
    type: str


class ModelsListResponse(BaseResponse):
    models: List[ModelInfo]


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

class TrainResponse(BaseResponse):
    pipeline_run: str
    states_trained: List[str]
    best_models: Dict[str, str]
    total_metrics_computed: int
    message: str = "Training pipeline completed successfully."


# ---------------------------------------------------------------------------
# All states forecast
# ---------------------------------------------------------------------------

class AllForecastsResponse(BaseResponse):
    forecasts: List[PredictResponse]
    total_states: int


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    detail: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

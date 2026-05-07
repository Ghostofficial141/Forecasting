"""
schemas/request.py
==================
Pydantic v2 request schemas for the forecasting REST API.
All fields include validation, examples, and clear docstrings.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request body for the POST /predict endpoint."""

    state: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the US state (must match dataset values exactly).",
        examples=["California", "Texas", "New York"],
    )
    weeks: int = Field(
        default=8,
        ge=1,
        le=52,
        description="Number of future weeks to forecast (1–52).",
        examples=[8],
    )
    model: Optional[Literal["SARIMA", "Prophet", "XGBoost", "LSTM"]] = Field(
        default=None,
        description=(
            "Force a specific model. If omitted, the best model for the "
            "state (from the last training run) is used automatically."
        ),
        examples=[None, "Prophet"],
    )

    @field_validator("state")
    @classmethod
    def state_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("state cannot be blank")
        return stripped

    model_config = {"json_schema_extra": {
        "example": {
            "state": "California",
            "weeks": 8,
            "model": None,
        }
    }}


class TrainRequest(BaseModel):
    """Optional request body for POST /train (all fields optional)."""

    force_retrain: bool = Field(
        default=False,
        description="If True, retrain even if artifacts already exist.",
    )

    model_config = {"json_schema_extra": {
        "example": {"force_retrain": False}
    }}

"""
exception.py
============
Custom exception hierarchy for the Sales Forecasting System.
Provides structured error context (module, function, line number) so logs
carry enough context to diagnose issues without reading stack traces manually.
"""

import sys
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_detail(error: Exception) -> str:
    """
    Extract detailed error information from the current exception context.

    Returns a formatted string with:
      - Python script name
      - Line number
      - Error message
    """
    _, _, exc_tb = sys.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return (
            f"Error occurred in script: [{file_name}] "
            f"at line number: [{line_number}] "
            f"with message: [{str(error)}]"
        )
    return str(error)


# ---------------------------------------------------------------------------
# Base Exception
# ---------------------------------------------------------------------------

class ForecastingSystemError(Exception):
    """
    Base exception for all forecasting system errors.
    Automatically logs and formats error details.
    """

    def __init__(self, message: str, error: Optional[Exception] = None):
        self.message = message
        if error is not None:
            self.detail = _error_detail(error)
        else:
            self.detail = message
        super().__init__(self.detail)
        logger.error(self.detail)

    def __str__(self) -> str:
        return self.detail


# ---------------------------------------------------------------------------
# Specific Exceptions
# ---------------------------------------------------------------------------

class DataIngestionError(ForecastingSystemError):
    """Raised when data cannot be read or loaded from the source."""
    pass


class DataValidationError(ForecastingSystemError):
    """Raised when the dataset fails schema or quality checks."""
    pass


class PreprocessingError(ForecastingSystemError):
    """Raised during data cleaning, imputation, or resampling steps."""
    pass


class FeatureEngineeringError(ForecastingSystemError):
    """Raised when feature construction fails (lags, rolling stats, etc.)."""
    pass


class ModelTrainingError(ForecastingSystemError):
    """Raised when a model fails to fit on training data."""
    pass


class ModelEvaluationError(ForecastingSystemError):
    """Raised when evaluation metrics cannot be computed."""
    pass


class ModelSelectionError(ForecastingSystemError):
    """Raised when no suitable model can be identified for a state."""
    pass


class PredictionError(ForecastingSystemError):
    """Raised when inference fails (model missing, invalid input, etc.)."""
    pass


class ConfigurationError(ForecastingSystemError):
    """Raised when a required configuration key is missing or invalid."""
    pass


class APIError(ForecastingSystemError):
    """Raised for REST API layer errors (input validation, serialisation)."""
    pass

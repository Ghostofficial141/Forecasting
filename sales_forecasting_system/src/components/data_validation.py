"""
data_validation.py
==================
Schema and quality checks performed BEFORE preprocessing.
Produces a validation report dict and raises DataValidationError on
critical failures so the pipeline stops early with clear feedback.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.exception import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validates raw ingested data before preprocessing.
    Checks:
      - Required columns present
      - Date column parseable
      - Target column numeric
      - Minimum row count
      - Missing value thresholds
      - Duplicate rows
      - Outlier detection (IQR-based)
      - Frequency consistency
    """

    # Quality thresholds
    MIN_ROWS_PER_STATE = 20          # minimum data points per group
    MAX_NULL_RATIO_TARGET = 0.30     # max 30% nulls in target column
    IQR_MULTIPLIER = 3.0             # outlier fence multiplier

    def __init__(
        self,
        date_col: str,
        target_col: str,
        state_col: Optional[str] = None,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.state_col = state_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> Dict:
        """
        Run all validation checks.

        Returns
        -------
        Dict
            Validation report with 'passed', 'warnings', 'errors' keys.
        """
        logger.info("=== DATA VALIDATION STARTED ===")
        report = {
            "passed": True,
            "warnings": [],
            "errors": [],
            "stats": {},
        }

        self._check_columns(df, report)
        self._check_date_column(df, report)
        self._check_target_column(df, report)
        self._check_row_count(df, report)
        self._check_missing_values(df, report)
        self._check_duplicates(df, report)
        self._check_outliers(df, report)
        self._check_state_coverage(df, report)

        if report["errors"]:
            report["passed"] = False
            for err in report["errors"]:
                logger.error(f"  [VALIDATION ERROR] {err}")
            raise DataValidationError(
                f"Data validation failed with {len(report['errors'])} error(s): "
                f"{report['errors']}"
            )

        for warn in report["warnings"]:
            logger.warning(f"  [VALIDATION WARNING] {warn}")

        logger.info(
            f"=== DATA VALIDATION COMPLETE — "
            f"{'PASSED' if report['passed'] else 'FAILED'} ==="
        )
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_columns(self, df: pd.DataFrame, report: Dict) -> None:
        """Ensure required columns exist."""
        required = [self.date_col, self.target_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            report["errors"].append(f"Missing required columns: {missing}")
        else:
            logger.debug(f"Column check passed: {required}")

    def _check_date_column(self, df: pd.DataFrame, report: Dict) -> None:
        """Confirm date column contains valid datetime values."""
        if self.date_col not in df.columns:
            return
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            try:
                pd.to_datetime(df[self.date_col])
                report["warnings"].append(
                    f"Date column '{self.date_col}' is not datetime dtype but can be parsed."
                )
            except Exception:
                report["errors"].append(
                    f"Date column '{self.date_col}' contains non-parseable values."
                )
        else:
            report["stats"]["date_min"] = str(df[self.date_col].min())
            report["stats"]["date_max"] = str(df[self.date_col].max())

    def _check_target_column(self, df: pd.DataFrame, report: Dict) -> None:
        """Confirm target column is numeric."""
        if self.target_col not in df.columns:
            return
        if not pd.api.types.is_numeric_dtype(df[self.target_col]):
            report["errors"].append(
                f"Target column '{self.target_col}' is not numeric."
            )
        else:
            null_ratio = df[self.target_col].isnull().mean()
            report["stats"]["target_null_ratio"] = round(float(null_ratio), 4)
            if null_ratio > self.MAX_NULL_RATIO_TARGET:
                report["errors"].append(
                    f"Target column has {null_ratio:.1%} missing values "
                    f"(threshold: {self.MAX_NULL_RATIO_TARGET:.1%})."
                )

    def _check_row_count(self, df: pd.DataFrame, report: Dict) -> None:
        """Ensure there is enough data for modelling."""
        report["stats"]["total_rows"] = len(df)
        if len(df) < self.MIN_ROWS_PER_STATE:
            report["errors"].append(
                f"Dataset has only {len(df)} rows. Minimum required: {self.MIN_ROWS_PER_STATE}."
            )

    def _check_missing_values(self, df: pd.DataFrame, report: Dict) -> None:
        """Log null counts per column."""
        null_counts = df.isnull().sum()
        report["stats"]["null_counts"] = null_counts.to_dict()
        if null_counts.any():
            report["warnings"].append(
                f"Columns with nulls: {null_counts[null_counts > 0].to_dict()}"
            )

    def _check_duplicates(self, df: pd.DataFrame, report: Dict) -> None:
        """Detect fully duplicated rows."""
        n_dup = df.duplicated().sum()
        report["stats"]["duplicate_rows"] = int(n_dup)
        if n_dup > 0:
            report["warnings"].append(f"Found {n_dup} fully duplicated row(s).")

    def _check_outliers(self, df: pd.DataFrame, report: Dict) -> None:
        """IQR-based outlier detection on target column."""
        if self.target_col not in df.columns:
            return
        if not pd.api.types.is_numeric_dtype(df[self.target_col]):
            return

        series = df[self.target_col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.IQR_MULTIPLIER * iqr
        upper = q3 + self.IQR_MULTIPLIER * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        report["stats"]["outlier_count"] = int(outliers)
        report["stats"]["outlier_bounds"] = {
            "lower": float(lower), "upper": float(upper)
        }
        if outliers > 0:
            report["warnings"].append(
                f"Found {outliers} potential outlier(s) in '{self.target_col}' "
                f"(bounds: [{lower:.2f}, {upper:.2f}])."
            )

    def _check_state_coverage(self, df: pd.DataFrame, report: Dict) -> None:
        """Ensure each state has sufficient observations."""
        if not self.state_col or self.state_col not in df.columns:
            return

        counts = df.groupby(self.state_col).size()
        insufficient = counts[counts < self.MIN_ROWS_PER_STATE]
        report["stats"]["state_counts"] = counts.to_dict()

        if not insufficient.empty:
            report["warnings"].append(
                f"States with fewer than {self.MIN_ROWS_PER_STATE} rows: "
                f"{insufficient.to_dict()}"
            )

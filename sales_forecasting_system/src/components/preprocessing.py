"""
preprocessing.py
================
Full preprocessing pipeline:
  1. Drop full duplicates
  2. Generate missing dates (gap fill)
  3. Interpolate / forward-fill missing target values
  4. Clip outliers (optional)
  5. Sort by [state, date]
  6. Validate resulting frequency
  7. Return clean DataFrame + scaler artefact
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.exception import PreprocessingError
from src.utils.helpers import (
    fill_missing_dates,
    infer_frequency,
    load_yaml,
    ensure_dir,
    save_pickle,
)
from src.utils.logger import get_logger
from src.constants import CONFIG_PATH, MODELS_DIR

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Cleans and prepares time-series data for feature engineering.
    Handles missing dates, missing values, outlier clipping, and scaling.
    """

    def __init__(
        self,
        date_col: str,
        target_col: str,
        state_col: Optional[str],
        config_path=None,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.state_col = state_col
        self.config = load_yaml(config_path or CONFIG_PATH)
        self.freq = self.config["data"].get("frequency", "W")
        self.scalers: Dict[str, MinMaxScaler] = {}  # one scaler per state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw ingested dataframe.

        Returns
        -------
        pd.DataFrame
            Clean, gap-filled, sorted dataframe ready for feature engineering.
        """
        logger.info("=== PREPROCESSING STARTED ===")

        df = self._drop_duplicates(df)
        df = self._ensure_datetime(df)
        df = self._fill_date_gaps(df)
        df = self._impute_missing_target(df)
        df = self._clip_outliers(df)
        df = self._sort(df)
        df = self._validate_frequency(df)

        logger.info(f"Preprocessing complete. Output shape: {df.shape}")
        logger.info("=== PREPROCESSING COMPLETE ===")
        return df

    def fit_scalers(self, df: pd.DataFrame) -> Dict[str, MinMaxScaler]:
        """
        Fit a MinMaxScaler per state on the target column.
        Scalers are saved to artifacts/models/ for inference use.
        """
        ensure_dir(MODELS_DIR)

        if self.state_col and self.state_col in df.columns:
            for state, grp in df.groupby(self.state_col):
                scaler = MinMaxScaler(feature_range=(0, 1))
                values = grp[self.target_col].values.reshape(-1, 1)
                scaler.fit(values)
                self.scalers[str(state)] = scaler
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df[self.target_col].values.reshape(-1, 1))
            self.scalers["__all__"] = scaler

        scaler_path = MODELS_DIR / "scalers.pkl"
        save_pickle(self.scalers, scaler_path)
        logger.info(f"Scalers saved to {scaler_path}")
        return self.scalers

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove fully duplicated rows."""
        n_before = len(df)
        df = df.drop_duplicates()
        dropped = n_before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} duplicate rows.")
        return df.reset_index(drop=True)

    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee date column is datetime64."""
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            df[self.date_col] = pd.to_datetime(df[self.date_col], infer_datetime_format=True)
        return df

    def _fill_date_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate a complete date grid and left-join existing data."""
        try:
            n_before = len(df)
            df = fill_missing_dates(
                df,
                date_col=self.date_col,
                freq=self.freq,
                group_col=self.state_col,
            )
            n_after = len(df)
            if n_after > n_before:
                logger.info(f"Date gaps filled: {n_after - n_before} synthetic rows added.")
            return df
        except Exception as e:
            raise PreprocessingError("Failed to fill date gaps", error=e)

    def _impute_missing_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing target values per state group.
        Strategy: linear interpolation, then forward-fill, then backward-fill
        to handle edge cases at series boundaries.
        """
        try:
            n_null_before = df[self.target_col].isnull().sum()

            if self.state_col and self.state_col in df.columns:
                def _impute_group(grp):
                    grp = grp.sort_values(self.date_col)
                    grp[self.target_col] = (
                        grp[self.target_col]
                        .interpolate(method="linear", limit_direction="both")
                        .ffill()
                        .bfill()
                    )
                    return grp

                df = (
                    df.groupby(self.state_col, group_keys=False)
                    .apply(_impute_group)
                    .reset_index(drop=True)
                )
            else:
                df = df.sort_values(self.date_col)
                df[self.target_col] = (
                    df[self.target_col]
                    .interpolate(method="linear", limit_direction="both")
                    .ffill()
                    .bfill()
                )

            n_null_after = df[self.target_col].isnull().sum()
            logger.info(
                f"Imputed missing values: {n_null_before} → {n_null_after} nulls in target."
            )
            return df
        except Exception as e:
            raise PreprocessingError("Target imputation failed", error=e)

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorise extreme values at the 1st and 99th percentile.
        Applied per state to respect each series' distribution.
        """
        try:
            def _clip_group(grp):
                lo = grp[self.target_col].quantile(0.01)
                hi = grp[self.target_col].quantile(0.99)
                grp[self.target_col] = grp[self.target_col].clip(lower=lo, upper=hi)
                return grp

            if self.state_col and self.state_col in df.columns:
                df = (
                    df.groupby(self.state_col, group_keys=False)
                    .apply(_clip_group)
                    .reset_index(drop=True)
                )
            else:
                df = _clip_group(df)

            return df
        except Exception as e:
            raise PreprocessingError("Outlier clipping failed", error=e)

    def _sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by [state, date]."""
        sort_cols = [self.state_col, self.date_col] if self.state_col else [self.date_col]
        sort_cols = [c for c in sort_cols if c and c in df.columns]
        return df.sort_values(sort_cols).reset_index(drop=True)

    def _validate_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log inferred frequency and warn if it differs from config."""
        if self.state_col and self.state_col in df.columns:
            sample = df[df[self.state_col] == df[self.state_col].iloc[0]]
        else:
            sample = df

        inferred = infer_frequency(sample, self.date_col)
        if inferred != self.freq:
            logger.warning(
                f"Inferred frequency '{inferred}' differs from configured '{self.freq}'. "
                "Using configured value."
            )
        else:
            logger.info(f"Data frequency validated: {self.freq}")
        return df

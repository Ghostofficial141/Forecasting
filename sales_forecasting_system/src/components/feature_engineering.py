"""
feature_engineering.py
=======================
Creates all time-series features required by ML models (XGBoost, LSTM):
  - Lag features  : lag_1, lag_7, lag_30
  - Rolling stats : rolling_mean_7/30, rolling_std_7/30
  - Calendar      : day_of_week, week_of_year, month, quarter
  - Binary flags  : is_weekend, holiday_flag
  
Stateful fitting is supported so the same parameters can be applied
at inference time to unseen data without information leakage.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

from src.utils.exception import FeatureEngineeringError
from src.utils.helpers import load_yaml
from src.utils.logger import get_logger
from src.constants import CONFIG_PATH

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Generates lag features, rolling statistics, and calendar features.
    Works per group (state) to avoid cross-contamination between series.
    """

    def __init__(
        self,
        date_col: str,
        target_col: str,
        state_col: Optional[str] = None,
        config_path=None,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.state_col = state_col
        cfg = load_yaml(config_path or CONFIG_PATH)
        feat_cfg = cfg.get("features", {})
        self.lag_periods: List[int] = feat_cfg.get("lag_periods", [1, 7, 30])
        self.rolling_windows: List[int] = feat_cfg.get("rolling_windows", [7, 30])
        self.include_holiday_flag: bool = feat_cfg.get("include_holiday_flag", True)
        self.holiday_country: str = feat_cfg.get("holiday_country", "US")
        self._holiday_set: Optional[set] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.

        Parameters
        ----------
        df : pd.DataFrame
            Clean preprocessed dataframe.
        drop_na : bool
            If True, drop rows where lag/rolling features are NaN
            (normal for training). Set False for inference rows.

        Returns
        -------
        pd.DataFrame
            DataFrame with all engineered features appended.
        """
        logger.info("=== FEATURE ENGINEERING STARTED ===")

        try:
            df = df.copy()
            df = self._ensure_datetime(df)

            if self.state_col and self.state_col in df.columns:
                df = (
                    df.groupby(self.state_col, group_keys=False)
                    .apply(self._engineer_group)
                    .reset_index(drop=True)
                )
            else:
                df = self._engineer_group(df)

            if drop_na:
                n_before = len(df)
                df = df.dropna().reset_index(drop=True)
                logger.info(
                    f"Dropped {n_before - len(df)} rows with NaN after feature engineering."
                )

            feature_cols = self._get_feature_columns(df)
            logger.info(f"Feature columns created ({len(feature_cols)}): {feature_cols}")
            logger.info(f"Final shape after feature engineering: {df.shape}")
            logger.info("=== FEATURE ENGINEERING COMPLETE ===")
            return df

        except Exception as e:
            raise FeatureEngineeringError("Feature engineering failed", error=e)

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return list of feature column names (excluding target & identifiers)."""
        return self._get_feature_columns(df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        return df

    def _engineer_group(self, grp: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature functions to a single state group."""
        grp = grp.sort_values(self.date_col).copy()
        grp = self._add_lag_features(grp)
        grp = self._add_rolling_features(grp)
        grp = self._add_calendar_features(grp)
        grp = self._add_holiday_flag(grp)
        return grp

    def _add_lag_features(self, grp: pd.DataFrame) -> pd.DataFrame:
        """Add lag_1, lag_7, lag_30 (or as configured)."""
        for lag in self.lag_periods:
            col_name = f"lag_{lag}"
            grp[col_name] = grp[self.target_col].shift(lag)
        return grp

    def _add_rolling_features(self, grp: pd.DataFrame) -> pd.DataFrame:
        """Add rolling mean and std for each configured window."""
        for window in self.rolling_windows:
            grp[f"rolling_mean_{window}"] = (
                grp[self.target_col]
                .shift(1)                         # shift 1 to avoid leakage
                .rolling(window=window, min_periods=1)
                .mean()
            )
            grp[f"rolling_std_{window}"] = (
                grp[self.target_col]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0)
            )
        return grp

    def _add_calendar_features(self, grp: pd.DataFrame) -> pd.DataFrame:
        """Add day-of-week, week-of-year, month, quarter, is_weekend."""
        dt = grp[self.date_col]
        grp["day_of_week"] = dt.dt.dayofweek       # 0=Monday, 6=Sunday
        grp["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        grp["month"] = dt.dt.month
        grp["quarter"] = dt.dt.quarter
        grp["year"] = dt.dt.year
        grp["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        return grp

    def _add_holiday_flag(self, grp: pd.DataFrame) -> pd.DataFrame:
        """Add binary holiday flag using the `holidays` library."""
        if not self.include_holiday_flag:
            grp["holiday_flag"] = 0
            return grp

        if not HOLIDAYS_AVAILABLE:
            logger.warning(
                "holidays library not installed. Setting holiday_flag=0. "
                "Install with: pip install holidays"
            )
            grp["holiday_flag"] = 0
            return grp

        if self._holiday_set is None:
            self._holiday_set = self._build_holiday_set(grp)

        grp["holiday_flag"] = grp[self.date_col].apply(
            lambda d: 1 if d.date() in self._holiday_set else 0
        )
        return grp

    def _build_holiday_set(self, grp: pd.DataFrame) -> set:
        """Build set of holiday dates covering the series date range."""
        years = grp[self.date_col].dt.year.unique().tolist()
        try:
            h = holidays_lib.country_holidays(self.holiday_country, years=years)
            return set(h.keys())
        except Exception as exc:
            logger.warning(f"Could not build holiday set: {exc}. Disabling holidays.")
            return set()

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return feature columns (exclude target, date, state, id-like cols)."""
        exclude = {
            self.date_col, self.target_col,
            self.state_col or "__none__",
        }
        return [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and c != "index"
        ]

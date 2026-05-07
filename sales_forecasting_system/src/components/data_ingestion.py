"""
data_ingestion.py
=================
Responsible for:
  1. Reading the raw Excel / CSV file
  2. Auto-detecting key columns (date, target, state)
  3. Basic type coercion
  4. Saving the validated raw dataframe to processed/ for downstream steps
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.constants import (
    CONFIG_PATH, RAW_DATA_DIR, PROCESSED_DATA_DIR
)
from src.utils.exception import DataIngestionError
from src.utils.helpers import (
    load_yaml,
    detect_date_column,
    detect_target_column,
    detect_state_column,
    ensure_dir,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """
    Loads raw data, auto-detects column roles, coerces types,
    and persists an interim Parquet snapshot.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config = load_yaml(config_path or CONFIG_PATH)
        self.data_cfg = self.config["data"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Execute the ingestion pipeline.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, str]]
            (raw dataframe, column mapping dict)
        """
        logger.info("=== DATA INGESTION STARTED ===")

        raw_path = Path(self.data_cfg["raw_data_path"])
        if not raw_path.is_absolute():
            raw_path = Path(__file__).resolve().parents[2] / raw_path

        logger.info(f"Loading data from: {raw_path}")
        df = self._load_file(raw_path)
        logger.info(f"Raw data shape: {df.shape}")

        # Auto-detect or use config-specified columns
        column_map = self._resolve_column_names(df)
        logger.info(f"Column mapping: {column_map}")

        # Type coercion
        df = self._coerce_types(df, column_map)

        # Print EDA summary to logs
        self._eda_summary(df, column_map)

        # Persist interim snapshot
        out_path = Path(self.data_cfg["processed_data_path"])
        if not out_path.is_absolute():
            out_path = Path(__file__).resolve().parents[2] / out_path
        ensure_dir(out_path.parent)
        df.to_parquet(out_path, index=False)
        logger.info(f"Raw snapshot saved to: {out_path}")

        logger.info("=== DATA INGESTION COMPLETE ===")
        return df, column_map

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> pd.DataFrame:
        """Read Excel or CSV into a DataFrame."""
        try:
            suffix = path.suffix.lower()
            if suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            elif suffix == ".csv":
                df = pd.read_csv(path)
            elif suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                raise DataIngestionError(
                    f"Unsupported file format: {suffix}. "
                    "Expected .xlsx, .xls, .csv, or .parquet."
                )
            # Strip whitespace from column names
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            raise DataIngestionError(
                f"Failed to load data from {path}", error=e
            )

    def _resolve_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Return dict with keys 'date', 'target', 'state' mapped to
        actual column names in the dataframe.
        """
        cfg = self.data_cfg
        date_col = cfg.get("date_column") or detect_date_column(df)
        target_col = cfg.get("target_column") or detect_target_column(df)
        state_col = cfg.get("state_column") or detect_state_column(df)

        # Validate columns exist
        for role, col in [("date", date_col), ("target", target_col)]:
            if col not in df.columns:
                raise DataIngestionError(
                    f"Detected {role} column '{col}' not found in dataframe. "
                    f"Available columns: {list(df.columns)}"
                )

        if state_col and state_col not in df.columns:
            logger.warning(
                f"Detected state column '{state_col}' not found. "
                "Treating data as single-series."
            )
            state_col = None

        return {
            "date": date_col,
            "target": target_col,
            "state": state_col,
        }

    def _coerce_types(
        self,
        df: pd.DataFrame,
        column_map: Dict[str, str],
    ) -> pd.DataFrame:
        """Convert date column to datetime and target to float."""
        try:
            date_col = column_map["date"]
            target_col = column_map["target"]

            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

            if column_map.get("state"):
                df[column_map["state"]] = df[column_map["state"]].astype(str).str.strip()

            return df
        except Exception as e:
            raise DataIngestionError("Type coercion failed", error=e)

    def _eda_summary(self, df: pd.DataFrame, column_map: Dict[str, str]) -> None:
        """Log a comprehensive EDA summary."""
        date_col = column_map["date"]
        target_col = column_map["target"]
        state_col = column_map.get("state")

        logger.info("─── EDA SUMMARY ───────────────────────────────────────")
        logger.info(f"  Total rows        : {len(df):,}")
        logger.info(f"  Total columns     : {df.shape[1]}")
        logger.info(f"  Date range        : {df[date_col].min()} → {df[date_col].max()}")
        logger.info(f"  Date column       : {date_col}")
        logger.info(f"  Target column     : {target_col}")
        logger.info(f"  State column      : {state_col or 'N/A (single series)'}")
        logger.info(f"  Null values       :\n{df.isnull().sum().to_string()}")
        logger.info(f"  Duplicate rows    : {df.duplicated().sum()}")

        if state_col:
            states = df[state_col].unique()
            logger.info(f"  Unique states     : {len(states)} — {list(states)}")

        # Target statistics
        stats = df[target_col].describe()
        logger.info(f"  Target stats      :\n{stats.to_string()}")
        logger.info("───────────────────────────────────────────────────────")

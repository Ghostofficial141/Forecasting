"""
helpers.py
==========
Generic utility functions reused across the forecasting pipeline:
  - File I/O (YAML, JSON, pickle, CSV)
  - Date/time helpers
  - DataFrame helpers
  - Model serialisation helpers
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Union[str, Path]) -> Dict:
    """Load and return a YAML file as a Python dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Loaded YAML from {path}")
    return config


def save_yaml(data: Dict, path: Union[str, Path]) -> None:
    """Serialize a dict to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.debug(f"Saved YAML to {path}")


def load_json(path: Union[str, Path]) -> Dict:
    """Load and return a JSON file as a Python dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.debug(f"Loaded JSON from {path}")
    return data


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Serialize data to a JSON file with optional indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_serializer)
    logger.debug(f"Saved JSON to {path}")


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for non-serializable types like numpy floats."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Serialize an arbitrary Python object with pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.debug(f"Saved pickle to {path}")


def load_pickle(path: Union[str, Path]) -> Any:
    """Load a pickled Python object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.debug(f"Loaded pickle from {path}")
    return obj


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def detect_date_column(df: pd.DataFrame) -> str:
    """
    Heuristically detect the date column from a dataframe.
    Prefers columns already parsed as datetime or whose name contains
    'date', 'week', 'time', 'period', 'ds'.
    """
    # Already datetime
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if datetime_cols:
        return datetime_cols[0]

    # Name-based heuristic
    candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ["date", "week", "time", "period", "ds", "day"]
    )]
    if candidates:
        return candidates[0]

    raise ValueError("Cannot auto-detect date column. Please specify it in config.yaml.")


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Heuristically detect the sales/target column.
    Prefers numeric columns whose name contains 'sale', 'revenue',
    'qty', 'quantity', 'amount', 'y'.
    """
    candidates = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ["sale", "revenue", "qty", "quantity", "amount", "y", "units"])
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if candidates:
        return candidates[0]

    # Fallback: first numeric column that isn't likely an ID
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_id_cols = [c for c in numeric_cols if "id" not in c.lower() and "index" not in c.lower()]
    if non_id_cols:
        return non_id_cols[0]

    raise ValueError("Cannot auto-detect target column. Please specify it in config.yaml.")


def detect_state_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the state/region/group column from the dataframe.
    Returns None if no clear candidate found (single-series case).
    """
    candidates = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ["state", "region", "city", "area", "location", "store", "group", "segment"])
        and not pd.api.types.is_numeric_dtype(df[c])
    ]
    return candidates[0] if candidates else None


def fill_missing_dates(
    df: pd.DataFrame,
    date_col: str,
    freq: str = "W",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a complete date range for each group and merge back.
    Fills gaps with NaN so downstream imputation can handle them.
    """
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    full_range = pd.date_range(start=min_date, end=max_date, freq=freq)

    if group_col is None:
        full_df = pd.DataFrame({date_col: full_range})
        df = full_df.merge(df, on=date_col, how="left")
        return df

    groups = df[group_col].unique()
    frames = []
    for grp in groups:
        full_df = pd.DataFrame({date_col: full_range, group_col: grp})
        grp_df = df[df[group_col] == grp]
        merged = full_df.merge(grp_df, on=[date_col, group_col], how="left")
        frames.append(merged)

    return pd.concat(frames, ignore_index=True)


def infer_frequency(df: pd.DataFrame, date_col: str) -> str:
    """
    Infer the dominant frequency of the time series.
    Returns pandas-compatible offset alias ('D', 'W', 'M', etc.).
    """
    sorted_dates = df[date_col].sort_values().drop_duplicates()
    diffs = sorted_dates.diff().dropna()
    median_diff = diffs.median()

    days = median_diff.days
    if days <= 1:
        return "D"
    elif days <= 7:
        return "W"
    elif days <= 15:
        return "2W"
    elif days <= 32:
        return "MS"
    elif days <= 95:
        return "QS"
    else:
        return "AS"


def get_timestamp_str() -> str:
    """Return a file-safe timestamp string like '20240101_153045'."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def split_time_series(
    df: pd.DataFrame,
    date_col: str,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-series-safe train/test split (no shuffling, chronological).
    
    Parameters
    ----------
    df : pd.DataFrame
        Sorted dataframe.
    date_col : str
        Name of the date column.
    test_ratio : float
        Fraction of data to use for testing.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

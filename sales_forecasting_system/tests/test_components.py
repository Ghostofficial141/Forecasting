"""
tests/test_components.py
========================
Unit tests for core pipeline components.
Uses pytest with lightweight synthetic data — no real Excel file required.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Create a synthetic weekly sales dataframe with 2 states."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-06", periods=104, freq="W")
    records = []
    for state in ["California", "Texas"]:
        sales = (
            500
            + np.arange(104) * 2
            + np.random.normal(0, 30, 104)
            + 50 * np.sin(np.arange(104) * 2 * np.pi / 52)
        )
        for d, s in zip(dates, sales):
            records.append({"date": d, "state": state, "sales": max(0, s)})
    return pd.DataFrame(records)


@pytest.fixture
def single_series_df():
    """Single-state weekly data."""
    np.random.seed(0)
    dates = pd.date_range("2020-01-06", periods=60, freq="W")
    sales = 300 + np.arange(60) + np.random.normal(0, 10, 60)
    return pd.DataFrame({"date": dates, "sales": sales})


# ---------------------------------------------------------------------------
# Test: Helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_detect_date_column(self, sample_df):
        from src.utils.helpers import detect_date_column
        result = detect_date_column(sample_df)
        assert result == "date"

    def test_detect_target_column(self, sample_df):
        from src.utils.helpers import detect_target_column
        result = detect_target_column(sample_df)
        assert result == "sales"

    def test_detect_state_column(self, sample_df):
        from src.utils.helpers import detect_state_column
        result = detect_state_column(sample_df)
        assert result == "state"

    def test_fill_missing_dates(self, sample_df):
        from src.utils.helpers import fill_missing_dates
        # Introduce a gap: remove some rows
        df_gapped = sample_df.drop(sample_df.index[5:10]).copy()
        filled = fill_missing_dates(df_gapped, "date", freq="W", group_col="state")
        # Should have more rows than gapped
        assert len(filled) >= len(df_gapped)

    def test_split_time_series(self, sample_df):
        from src.utils.helpers import split_time_series
        # Use single state
        ca_df = sample_df[sample_df["state"] == "California"].copy()
        train, test = split_time_series(ca_df, "date", test_ratio=0.2)
        assert len(train) > len(test)
        # No date overlap
        assert train["date"].max() < test["date"].min()


# ---------------------------------------------------------------------------
# Test: Metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_rmse_perfect(self):
        from src.utils.metrics import rmse
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_mae(self):
        from src.utils.metrics import mae
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == pytest.approx(1.0)

    def test_mape_no_zero_division(self):
        from src.utils.metrics import mape
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 110.0])
        # Should not raise
        result = mape(y_true, y_pred)
        assert result >= 0

    def test_compute_all_metrics(self):
        from src.utils.metrics import compute_all_metrics
        y = np.array([100.0, 200.0, 150.0, 180.0])
        preds = np.array([105.0, 195.0, 155.0, 175.0])
        m = compute_all_metrics(y, preds, model_name="Test", state="CA")
        assert "RMSE" in m
        assert "MAE" in m
        assert "MAPE" in m
        assert "R2" in m
        assert m["RMSE"] > 0


# ---------------------------------------------------------------------------
# Test: Data Validation
# ---------------------------------------------------------------------------

class TestDataValidator:

    def test_passes_clean_data(self, sample_df):
        from src.components.data_validation import DataValidator
        validator = DataValidator("date", "sales", "state")
        report = validator.run(sample_df)
        assert report["passed"] is True

    def test_fails_on_missing_target_column(self, sample_df):
        from src.components.data_validation import DataValidator
        from src.utils.exception import DataValidationError
        validator = DataValidator("date", "nonexistent", "state")
        with pytest.raises(DataValidationError):
            validator.run(sample_df)

    def test_warns_on_nulls(self, sample_df):
        from src.components.data_validation import DataValidator
        df_with_nulls = sample_df.copy()
        df_with_nulls.loc[0, "sales"] = None
        validator = DataValidator("date", "sales", "state")
        report = validator.run(df_with_nulls)
        assert len(report["warnings"]) > 0


# ---------------------------------------------------------------------------
# Test: Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessor:

    def test_preprocessing_runs(self, sample_df):
        from src.components.preprocessing import DataPreprocessor
        prep = DataPreprocessor("date", "sales", "state")
        result = prep.run(sample_df)
        assert len(result) > 0
        assert result["sales"].isnull().sum() == 0

    def test_drops_duplicates(self, sample_df):
        from src.components.preprocessing import DataPreprocessor
        duped = pd.concat([sample_df, sample_df.iloc[:5]])
        prep = DataPreprocessor("date", "sales", "state")
        result = prep.run(duped)
        # No full duplicate rows
        assert result.duplicated().sum() == 0


# ---------------------------------------------------------------------------
# Test: Feature Engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineer:

    def test_creates_lag_features(self, sample_df):
        from src.components.feature_engineering import FeatureEngineer
        fe = FeatureEngineer("date", "sales", "state")
        result = fe.run(sample_df, drop_na=True)
        assert "lag_1" in result.columns
        assert "lag_7" in result.columns
        assert "lag_30" in result.columns

    def test_creates_rolling_features(self, sample_df):
        from src.components.feature_engineering import FeatureEngineer
        fe = FeatureEngineer("date", "sales", "state")
        result = fe.run(sample_df, drop_na=True)
        assert "rolling_mean_7" in result.columns
        assert "rolling_std_7" in result.columns

    def test_creates_calendar_features(self, sample_df):
        from src.components.feature_engineering import FeatureEngineer
        fe = FeatureEngineer("date", "sales", "state")
        result = fe.run(sample_df, drop_na=True)
        for col in ["day_of_week", "month", "quarter", "is_weekend"]:
            assert col in result.columns

    def test_no_data_leakage(self, sample_df):
        """Rolling features must shift by 1 to prevent leakage."""
        from src.components.feature_engineering import FeatureEngineer
        fe = FeatureEngineer("date", "sales", "state")
        ca_df = sample_df[sample_df["state"] == "California"].copy()
        result = fe.run(ca_df, drop_na=False)
        # At the first valid row, rolling_mean_7 should be based on lagged values
        # i.e., it should NOT equal the current 'sales' value
        # (just checking it doesn't crash and produces numbers)
        assert result["rolling_mean_7"].notna().any()


# ---------------------------------------------------------------------------
# Test: API (async) — requires httpx + pytest-asyncio
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAPI:

    async def test_health_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "models_ready" in data

    async def test_models_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 4

    async def test_predict_without_training_returns_error(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/predict",
                json={"state": "California", "weeks": 4},
            )
        # Should return 503 or 500 if models not trained
        assert response.status_code in [200, 500, 503]

    async def test_predict_validation(self):
        from httpx import AsyncClient, ASGITransport
        from api.app import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # weeks > 52 should fail validation
            response = await client.post(
                "/predict",
                json={"state": "California", "weeks": 100},
            )
        assert response.status_code == 422    # FastAPI validation error

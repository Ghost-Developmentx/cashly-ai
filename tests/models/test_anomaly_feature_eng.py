import pandas as pd
import numpy as np

from app.models.anomaly.feature_engineering import (
    TransactionFeatureExtractor,
    BehaviorFeatureExtractor,
)


def _create_transactions(num_rows: int = 10) -> pd.DataFrame:
    """Create simple transactions DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=num_rows, freq="D")
    data = {
        "date": dates,
        "amount": np.linspace(10, 100, num_rows),
        "category": ["A", "B"] * (num_rows // 2) + ["A"] * (num_rows % 2),
    }
    return pd.DataFrame(data)


def test_transaction_feature_extractor_basic():
    df = _create_transactions(5)
    extractor = TransactionFeatureExtractor()
    extractor.fit(df)
    result = extractor.transform(df)

    expected_cols = [
        "amount_zscore",
        "amount_iqr_score",
        "amount_log",
        "hour",
        "day_of_week",
        "day_of_month",
        "is_weekend",
        "is_night",
        "category_deviation",
    ]
    for col in expected_cols:
        assert col in result.columns


def test_behavior_feature_extractor_basic():
    df = _create_transactions(10)

    trans_ext = TransactionFeatureExtractor()
    df = trans_ext.fit_transform(df)

    beh_ext = BehaviorFeatureExtractor()
    beh_ext.fit(df)
    result = beh_ext.transform(df)

    expected_cols = [
        "days_since_last",
        "daily_transaction_count",
        "trans_count_1d",
        "amount_mean_1d",
        "amount_std_1d",
        "trans_count_7d",
        "amount_mean_7d",
        "amount_std_7d",
        "amount_velocity",
        "amount_acceleration",
        "breaks_weekly_pattern",
    ]
    for col in expected_cols:
        assert col in result.columns
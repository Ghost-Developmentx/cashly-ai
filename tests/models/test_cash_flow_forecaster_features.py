import pandas as pd
import numpy as np
import pytest

from app.models.forecasting.cash_flow_forecaster import CashFlowForecaster

class DummyScaler:
    def fit(self, X):
        return self
    def transform(self, X):
        return X

class DummyModel:
    def predict(self, X):
        return np.ones(len(X))


def _simple_forecaster():
    forecaster = CashFlowForecaster(method="linear")
    # reduce feature complexity for test clarity
    forecaster.time_extractor.lag_days = [1, 2]
    forecaster.time_extractor.window_sizes = [3]
    return forecaster


def _sample_data():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "amount": [1, 2, 3, 4, 5]
    })


def test_create_forecast_features():
    forecaster = _simple_forecaster()
    data = _sample_data()

    processed = forecaster.preprocess(data)
    features = forecaster.extract_features(processed)

    target_date = pd.Timestamp("2024-01-06")
    feature_row = forecaster._create_forecast_features(features, target_date)

    assert feature_row["lag_1"] == 5
    assert feature_row["lag_2"] == 4
    assert feature_row["rolling_mean_3"] == pytest.approx(4.0)
    assert feature_row["rolling_min_3"] == 3
    assert feature_row["rolling_max_3"] == 5


def test_generate_forecast_length():
    forecaster = _simple_forecaster()
    data = _sample_data()

    processed = forecaster.preprocess(data)
    feature_data = forecaster.extract_features(processed)

    # attach dummy components
    forecaster.scaler = DummyScaler()
    forecaster.models = {"linear": DummyModel()}
    forecaster.method = "linear"
    forecaster.feature_names = list(feature_data.columns)

    last_date = data["date"].iloc[-1]
    forecast = forecaster._generate_forecast(feature_data, last_date, horizon=2)

    assert len(forecast) == 2
    assert list(forecast["date"]) == [last_date + pd.Timedelta(days=1), last_date + pd.Timedelta(days=2)]
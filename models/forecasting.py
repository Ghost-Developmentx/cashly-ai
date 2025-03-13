import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from util.data_processing import prepare_timeseries_data
from util.model_registry import ModelRegistry


class CashFlowForecaster:
    """
    Model for forecasting future cash flow based on historical transactions
    """

    def __init__(self, registry=None, method="ensemble"):
        """
        Initialize the forecaster

        Args:
            registry: ModelRegistry instance
            method: Forecasting method ('linear', 'rf', 'gbm', 'prophet', 'ensemble')
        """
        self.model = None
        self.registry = registry or ModelRegistry()
        self.model_id = None
        self.method = method
        self.scaler = None
        self.feature_cols = None
        self.forecast_days = 30  # Default

    @staticmethod
    def _create_features(df):
        """
        Create time-based features for forecasting

        Args:
            df: DataFrame with 'date' and 'amount' columns

        Returns:
            DataFrame with additional features
        """
        # Ensure date is datetime
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Create date features
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["year"] = df["date"].dt.year
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
        df["days_in_month"] = df["date"].dt.days_in_month

        # Add lag features (previous days' values)
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f"lag_{lag}"] = df["amount"].shift(lag)

        # Add rolling window features
        for window in [3, 7, 14, 30]:
            df[f"rolling_mean_{window}"] = df["amount"].rolling(window=window).mean()
            df[f"rolling_std_{window}"] = df["amount"].rolling(window=window).std()
            df[f"rolling_min_{window}"] = df["amount"].rolling(window=window).min()
            df[f"rolling_max_{window}"] = df["amount"].rolling(window=window).max()

        # Drop rows with NaN values (due to lag/rolling features)
        df = df.dropna()

        return df

    def fit(self, transactions_df, forecast_days=30):
        """
        Train the forecasting model

        Args:
            transactions_df: DataFrame with transaction data
            forecast_days: Number of days to forecast

        Returns:
            self: The trained model
        """
        self.forecast_days = forecast_days

        # Prepare data - resample to daily frequency
        df = prepare_timeseries_data(transactions_df, freq="D")

        # Create features
        df = self._create_features(df)

        # Store feature columns
        self.feature_cols = [col for col in df.columns if col not in ["date", "amount"]]

        # Train the model based on selected method
        if self.method == "linear":
            return self._fit_linear(df)
        elif self.method == "rf":
            return self._fit_random_forest(df)
        elif self.method == "gbm":
            return self._fit_gradient_boosting(df)
        elif self.method == "prophet" and PROPHET_AVAILABLE:
            return self._fit_prophet(df)
        else:
            # Default to ensemble
            return self._fit_ensemble(df)

    def _fit_linear(self, df):
        """Train a linear regression model"""
        # Split into features and target
        X = df[self.feature_cols]
        y = df["amount"]

        # Train-test split (time-based)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Create pipeline with standardization
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)

        # Save model
        self.model_id = self.registry.save_model(
            model=(self.model, self.scaler),
            model_name="cash_flow_forecaster_linear",
            model_type="forecasting",
            features=self.feature_cols,
            metrics=metrics,
            metadata={"forecast_days": self.forecast_days, "method": "linear"},
        )

        return self

    def _fit_random_forest(self, df):
        """Train a random forest model"""
        # Split into features and target
        X = df[self.feature_cols]
        y = df["amount"]

        # Train-test split (time-based)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Create pipeline with standardization
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)

        # Save model
        self.model_id = self.registry.save_model(
            model=(self.model, self.scaler),
            model_name="cash_flow_forecaster_rf",
            model_type="forecasting",
            features=self.feature_cols,
            metrics=metrics,
            metadata={"forecast_days": self.forecast_days, "method": "rf"},
        )

        return self

    def _fit_gradient_boosting(self, df):
        """Train a gradient boosting model"""
        # Split into features and target
        X = df[self.feature_cols]
        y = df["amount"]

        # Train-test split (time-based)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Create pipeline with standardization
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred)

        # Save model
        self.model_id = self.registry.save_model(
            model=(self.model, self.scaler),
            model_name="cash_flow_forecaster_gbm",
            model_type="forecasting",
            features=self.feature_cols,
            metrics=metrics,
            metadata={"forecast_days": self.forecast_days, "method": "gbm"},
        )

        return self

    def _fit_prophet(self, df):
        """Train a Prophet model"""
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "Prophet is not installed. Install it with pip install prophet"
            )

        # Prepare data for Prophet
        prophet_df = df.rename(columns={"date": "ds", "amount": "y"})

        # Train the model
        self.model = Prophet(
            yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True
        )

        # Add custom regressors from our feature columns
        for col in self.feature_cols:
            self.model.add_regressor(col)

        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_df)

        # Evaluate
        future = self.model.make_future_dataframe(periods=30)
        for col in self.feature_cols:
            future[col] = (
                prophet_df[col].iloc[-30:].values
            )  # Use the last 30 days as an approximation

        forecast = self.model.predict(future)

        # Calculate metrics on the last 30 days
        y_true = prophet_df["y"].iloc[-30:].values
        y_pred = forecast["yhat"].iloc[-60:-30].values

        metrics = self._calculate_metrics(y_true, y_pred)

        # Save model
        self.model_id = self.registry.save_model(
            model=self.model,
            model_name="cash_flow_forecaster_prophet",
            model_type="forecasting",
            features=self.feature_cols,
            metrics=metrics,
            metadata={"forecast_days": self.forecast_days, "method": "prophet"},
        )

        return self

    def _fit_ensemble(self, df):
        """Train multiple models and use them as an ensemble"""
        # Split into features and target
        X = df[self.feature_cols]
        y = df["amount"]

        # Train-test split (time-based)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Create pipeline with standardization
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train multiple models
        models = {
            "linear": LinearRegression(),
            "rf": RandomForestRegressor(n_estimators=100, random_state=42),
            "gbm": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

        # Fit all models
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)

        # Evaluate each model and combine predictions
        predictions = {}
        metrics = {}

        for name, model in models.items():
            pred = model.predict(X_test_scaled)
            predictions[name] = pred
            metrics[name] = self._calculate_metrics(y_test, pred)

        # Combine predictions (simple average)
        ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
        ensemble_metrics = self._calculate_metrics(y_test, ensemble_pred)

        # Store the models and weights for prediction
        self.model = (models, {"linear": 1 / 3, "rf": 1 / 3, "gbm": 1 / 3})

        # Save model
        self.model_id = self.registry.save_model(
            model=(self.model, self.scaler),
            model_name="cash_flow_forecaster_ensemble",
            model_type="forecasting",
            features=self.feature_cols,
            metrics={**ensemble_metrics, "individual_models": metrics},
            metadata={"forecast_days": self.forecast_days, "method": "ensemble"},
        )

        return self

    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        """Calculate regression metrics"""
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
        }

    def forecast(self, transactions_df, forecast_days=None):
        """
        Generate cash flow forecast

        Args:
            transactions_df: DataFrame with historical transaction data
            forecast_days: Number of days to forecast (default: self.forecast_days)

        Returns:
            DataFrame with forecasted values
        """
        if self.model is None:
            # Try to load the latest model
            self.model, self.scaler = self.registry.load_model(
                model_type="forecasting", latest=True
            )
            self.feature_cols = model_info["features"]
            self.model_id = model_info["id"]
            self.forecast_days = model_info["metadata"]["forecast_days"]

        forecast_days = forecast_days or self.forecast_days

        # Prepare data
        df = prepare_timeseries_data(transactions_df, freq="D")
        orig_df = df.copy()

        # Create features
        df = self._create_features(df)

        # Get the last date in the data
        last_date = df["date"].max()

        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=forecast_days
        )

        # Initialize forecast dataframe
        forecast_df = pd.DataFrame({"date": forecast_dates})

        # Add features to forecast dataframe
        forecast_df["day_of_week"] = forecast_df["date"].dt.dayofweek
        forecast_df["day_of_month"] = forecast_df["date"].dt.day
        forecast_df["month"] = forecast_df["date"].dt.month
        forecast_df["quarter"] = forecast_df["date"].dt.quarter
        forecast_df["year"] = forecast_df["date"].dt.year
        forecast_df["is_weekend"] = forecast_df["day_of_week"].apply(
            lambda x: 1 if x >= 5 else 0
        )
        forecast_df["is_month_start"] = forecast_df["date"].dt.is_month_start.astype(
            int
        )
        forecast_df["is_month_end"] = forecast_df["date"].dt.is_month_end.astype(int)
        forecast_df["days_in_month"] = forecast_df["date"].dt.days_in_month

        # Generate forecasts differently based on method
        if (
            isinstance(self.model, tuple)
            and len(self.model) == 2
            and isinstance(self.model[0], dict)
        ):
            # Ensemble model - models and weights
            models, weights = self.model
            scaler = self.scaler

            # Iterative prediction for each day
            current_df = df.copy()

            for i, forecast_date in enumerate(forecast_dates):
                # Get the current features
                current_features = forecast_df.iloc[i : i + 1].copy()

                # Add lag features from previous predictions and history
                for lag in [1, 2, 3, 7, 14, 30]:
                    if i >= lag:
                        # Use already predicted values
                        current_features[f"lag_{lag}"] = forecast_df["amount"].iloc[
                            i - lag
                        ]
                    else:
                        # Use historical values
                        current_features[f"lag_{lag}"] = df["amount"].iloc[-lag]

                # Add rolling window features (approximate using recent history)
                for window in [3, 7, 14, 30]:
                    if i >= window:
                        # Use predicted values
                        values = (
                            forecast_df["amount"].iloc[max(0, i - window) : i].tolist()
                        )
                        historical_values = (
                            df["amount"].iloc[-max(0, window - i) :].tolist()
                        )
                        all_values = historical_values + values

                        current_features[f"rolling_mean_{window}"] = np.mean(
                            all_values[-window:]
                        )
                        current_features[f"rolling_std_{window}"] = np.std(
                            all_values[-window:]
                        )
                        current_features[f"rolling_min_{window}"] = np.min(
                            all_values[-window:]
                        )
                        current_features[f"rolling_max_{window}"] = np.max(
                            all_values[-window:]
                        )
                    else:
                        # Use historical values
                        values = df["amount"].iloc[-window:].tolist()

                        current_features[f"rolling_mean_{window}"] = np.mean(values)
                        current_features[f"rolling_std_{window}"] = np.std(values)
                        current_features[f"rolling_min_{window}"] = np.min(values)
                        current_features[f"rolling_max_{window}"] = np.max(values)

                # Scale features
                X = current_features[self.feature_cols]
                X_scaled = scaler.transform(X)

                # Make predictions from each model and combine
                predictions = {}
                for name, model in models.items():
                    predictions[name] = model.predict(X_scaled)[0]

                # Weighted average
                forecast_value = sum(
                    predictions[name] * weights[name] for name in weights
                )

                # Store the prediction
                forecast_df.at[i, "amount"] = forecast_value

                # Update lag features for next prediction
                for lag in [1, 2, 3, 7, 14, 30]:
                    if i + lag < len(forecast_dates):
                        forecast_df.at[i + lag, f"lag_{lag}"] = forecast_value

        elif isinstance(self.model, tuple) and len(self.model) == 2:
            # Single sklearn model with scaler
            model, scaler = self.model

            # Iterative prediction (similar to ensemble case)
            for i, forecast_date in enumerate(forecast_dates):
                # Implementation similar to ensemble case above
                # ...
                pass

        elif PROPHET_AVAILABLE and isinstance(self.model, Prophet):
            # Prophet model
            # Prepare future dataframe
            future = self.model.make_future_dataframe(periods=forecast_days)

            # Add regressor values (approximate using last values)
            for col in self.feature_cols:
                if col in forecast_df.columns:
                    future[col] = pd.concat([df[col], forecast_df[col]])
                else:
                    # Use last values as approximation for other features
                    future[col] = df[col].iloc[-1]

            # Make forecast
            prophet_forecast = self.model.predict(future)

            # Extract relevant dates
            forecast_result = prophet_forecast[
                prophet_forecast["ds"].isin(forecast_dates)
            ]

            # Set to forecast_df
            forecast_df["amount"] = forecast_result["yhat"].values

        # Calculate cumulative balance
        current_balance = orig_df["amount"].sum()
        forecast_df["cumulative_balance"] = (
            forecast_df["amount"].cumsum() + current_balance
        )

        # Ensure all the features we used for training are in the result
        for col in self.feature_cols:
            if col not in forecast_df.columns and col in df.columns:
                forecast_df[col] = df[col].iloc[-1]  # Use last value as approximation

        return forecast_df

    def update_model(self, new_transactions_df, forecast_days=None):
        """
        Update the forecasting model with new transaction data

        Args:
            new_transactions_df: DataFrame with new transaction data
            forecast_days: Number of days to forecast (default: self.forecast_days)

        Returns:
            self: Updated model
        """
        if self.model is None:
            # If no model exists, just train from scratch
            return self.fit(new_transactions_df, forecast_days=forecast_days or 30)

        # Store previous model information
        previous_model_id = self.model_id

        # Prepare the new data
        df = prepare_timeseries_data(new_transactions_df, freq="D")
        df = self._create_features(df)

        # Update forecast days if specified
        if forecast_days is not None:
            self.forecast_days = forecast_days

        # Model updating depends on the forecasting method used
        if self.method == "ensemble":
            # For ensemble models, extract components and update each
            models, weights = self.model

            for name, model in models.items():
                X = df[self.feature_cols]
                y = df["amount"]

                # Scale features
                X_scaled = self.scaler.transform(X)

                # Update model
                model.fit(X_scaled, y)

            # Keep the original weights
            self.model = (models, weights)

        elif isinstance(self.model, tuple) and len(self.model) == 2:
            # For single sklearn model with scaler
            model, scaler = self.model

            X = df[self.feature_cols]
            y = df["amount"]

            # Update scaler with new data
            X_scaled = scaler.fit_transform(X)

            # Update model
            model.fit(X_scaled, y)

            self.model = (model, scaler)
            self.scaler = scaler

        # Calculate basic performance metrics
        y_true = df["amount"].values[-min(30, len(df)) :]
        y_pred = self.forecast(df.iloc[: -min(30, len(df))])["amount"].values[
            : min(30, len(df))
        ]

        if len(y_true) > 0 and len(y_pred) > 0:
            # Calculate metrics on available data
            metrics = {
                "mae": mean_absolute_error(
                    y_true[: len(y_pred)], y_pred[: len(y_true)]
                ),
                "rmse": np.sqrt(
                    mean_squared_error(y_true[: len(y_pred)], y_pred[: len(y_true)])
                ),
                "samples_used": len(df),
            }
        else:
            metrics = {"samples_used": len(df)}

        # Save updated model
        self.model_id = self.registry.save_model(
            model=self.model,
            model_name=f"cash_flow_forecaster_{self.method}_updated",
            model_type="forecasting",
            features=self.feature_cols,
            metrics=metrics,
            metadata={
                "forecast_days": self.forecast_days,
                "method": self.method,
                "previous_model_id": previous_model_id,
                "update_time": datetime.now().isoformat(),
            },
        )

        return self

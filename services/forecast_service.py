import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.forecasting import CashFlowForecaster
from util.data_processing import prepare_timeseries_data
from util.model_registry import ModelRegistry


class ForecastService:
    """
    Service for forecasting cash flow and other financial metrics
    """

    def __init__(self):
        self.registry = ModelRegistry()
        self.forecaster = CashFlowForecaster(registry=self.registry)

        # Try to load existing model
        try:
            model_tuple, model_info = self.registry.load_model(
                model_type="forecasting", latest=True
            )

            self.forecaster.model, self.forecaster.scaler = model_tuple
            self.forecaster.feature_cols = model_info["features"]
            self.forecaster.model_id = model_info["id"]
            self.forecaster.forecast_days = model_info["metadata"]["forecast_days"]
            print(f"Loaded forecasting model: {model_info['id']}")
        except Exception as e:
            print(f"No existing forecasting model found: {str(e)}")

    def forecast_cash_flow(self, user_id, transactions, forecast_days=30):
        """
        Generate a cash flow forecast

        Args:
            user_id: User ID
            transactions: List of transaction dictionaries
            forecast_days: Number of days to forecast

        Returns:
            dict: Forecast results
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Ensure required columns exist
        required_cols = ["date", "amount"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Calculate current balance
        current_balance = df["amount"].sum()

        # If forecaster model doesn't exist, train a new one
        if self.forecaster.model is None:
            try:
                self.forecaster.fit(df, forecast_days=forecast_days)
            except Exception as e:
                print(f"Error training forecasting model: {str(e)}")
                # Fall back to simple forecasting method
                return self._simple_forecast(df, current_balance, forecast_days)

        # Generate forecast
        try:
            forecast_df = self.forecaster.forecast(df, forecast_days=forecast_days)

            # Format forecast for response
            forecast = [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "projected_amount": round(amount, 2),
                    "projected_balance": round(balance, 2),
                }
                for date, amount, balance in zip(
                    forecast_df["date"],
                    forecast_df["amount"],
                    forecast_df["cumulative_balance"],
                )
            ]

            # Calculate insights
            avg_daily_amount = forecast_df["amount"].mean()
            final_balance = forecast_df["cumulative_balance"].iloc[-1]

            insights = {
                "average_daily_amount": round(avg_daily_amount, 2),
                "current_balance": round(current_balance, 2),
                "projected_balance": round(final_balance, 2),
                "cash_flow_trend": "positive" if avg_daily_amount > 0 else "negative",
                "model_confidence": self._calculate_confidence(forecast_df),
            }

            return {"forecast": forecast, "insights": insights}

        except Exception as e:
            print(f"Error generating forecast: {str(e)}")
            # Fall back to simple forecasting method
            return self._simple_forecast(df, current_balance, forecast_days)

    def _calculate_confidence(self, forecast_df):
        """Calculate a confidence score for the forecast"""
        # This is a simple heuristic - in a real system we'd use proper
        # statistical methods to calculate prediction intervals

        # Check if we have enough data
        if len(forecast_df) < 10:
            return "low"

        # Check variability in predictions
        std_dev = forecast_df["amount"].std()
        mean_abs = forecast_df["amount"].abs().mean()

        if mean_abs == 0:
            coefficient_of_variation = 0
        else:
            coefficient_of_variation = std_dev / mean_abs

        if coefficient_of_variation > 1.0:
            return "low"
        elif coefficient_of_variation > 0.5:
            return "medium"
        else:
            return "high"

    def _simple_forecast(self, df, current_balance, forecast_days):
        """
        Simple rule-based forecasting as a fallback

        Args:
            df: DataFrame with transaction data
            current_balance: Current account balance
            forecast_days: Number of days to forecast

        Returns:
            dict: Forecast results
        """
        # Calculate average daily net change
        daily_df = prepare_timeseries_data(df, freq="D")

        avg_daily_net = daily_df["amount"].mean()
        std_dev = daily_df["amount"].std() if len(daily_df) > 1 else avg_daily_net * 0.1

        # Generate forecast dates
        last_date = df["date"].max()
        forecast_dates = [
            last_date + timedelta(days=i + 1) for i in range(forecast_days)
        ]

        # Generate forecast values with some randomness
        forecast_values = [
            avg_daily_net + np.random.normal(0, std_dev) for _ in range(forecast_days)
        ]

        # Calculate cumulative balance
        cumulative_forecast = [current_balance]
        running_total = current_balance

        for value in forecast_values:
            running_total += value
            cumulative_forecast.append(running_total)

        # Remove the first element (current balance)
        cumulative_forecast = cumulative_forecast[1:]

        # Format forecast for response
        forecast = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "projected_amount": round(amount, 2),
                "projected_balance": round(balance, 2),
            }
            for date, amount, balance in zip(
                forecast_dates, forecast_values, cumulative_forecast
            )
        ]

        # Calculate insights
        insights = {
            "average_daily_amount": round(avg_daily_net, 2),
            "current_balance": round(current_balance, 2),
            "projected_balance": round(cumulative_forecast[-1], 2),
            "cash_flow_trend": "positive" if avg_daily_net > 0 else "negative",
            "model_confidence": "low",  # Always low for the simple model
        }

        return {"forecast": forecast, "insights": insights}

    def train_forecasting_model(
        self, transactions_data, forecast_days=30, method="ensemble"
    ):
        """
        Train a new forecasting model

        Args:
            transactions_data: List of transaction dictionaries
            forecast_days: Number of days to forecast
            method: Forecasting method

        Returns:
            dict: Training results
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions_data)

        # Ensure required columns exist
        required_cols = ["date", "amount"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Create a new forecaster with the specified method
        self.forecaster = CashFlowForecaster(registry=self.registry, method=method)

        # Train the model
        try:
            self.forecaster.fit(df, forecast_days=forecast_days)

            # Get model metrics
            model_info = next(
                (
                    m
                    for m in self.registry.list_models("forecasting")
                    if m["id"] == self.forecaster.model_id
                ),
                {},
            )

            return {
                "success": True,
                "model_id": self.forecaster.model_id,
                "method": method,
                "metrics": model_info.get("metrics", {}),
                "message": f"Model trained successfully using {method} method",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_forecasting_model(
        self, transactions_data, forecast_days=None, method=None
    ):
        """
        Update the forecasting model with new transaction data

        Args:
            transactions_data: List of transaction dictionaries
            forecast_days: Number of days to forecast (optional)
            method: Forecasting method to use (optional - defaults to current method)

        Returns:
            dict: Update results
        """
        # Convert to DataFrame
        df = pd.DataFrame(transactions_data)

        # Ensure required columns exist
        required_cols = ["date", "amount"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # If forecaster model doesn't exist, train a new one
        if self.forecaster.model is None:
            try:
                self.forecaster.fit(df, forecast_days=forecast_days or 30)
                return {
                    "success": True,
                    "model_id": self.forecaster.model_id,
                    "message": "New forecasting model trained successfully",
                }
            except Exception as e:
                print(f"Error training forecasting model: {str(e)}")
                return {"success": False, "error": str(e)}

        # Update existing model
        try:
            # If method is specified and different from current, create a new model
            if method and method != self.forecaster.method:
                self.forecaster = CashFlowForecaster(
                    registry=self.registry, method=method
                )
                self.forecaster.fit(df, forecast_days=forecast_days or 30)
                message = f"Created new forecasting model with {method} method"
            else:
                # Update existing model
                self.forecaster.update_model(df, forecast_days=forecast_days)
                message = "Updated existing forecasting model"

            # Get updated metrics
            model_info = next(
                (
                    m
                    for m in self.registry.list_models("forecasting")
                    if m["id"] == self.forecaster.model_id
                ),
                {},
            )

            return {
                "success": True,
                "model_id": self.forecaster.model_id,
                "method": self.forecaster.method,
                "metrics": model_info.get("metrics", {}),
                "message": message,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

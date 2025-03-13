import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import os
import sys
from models.forecasting import CashFlowForecaster
from services.forecast_service import ForecastService


def generate_test_timeseries(
    days=365, start_date=None, seasonality=True, trend=True, noise_level=0.2
):
    """Generate synthetic cash flow data with trend and seasonality"""

    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)

    dates = [start_date + timedelta(days=i) for i in range(days)]
    amounts = []

    # Base level
    base = 100

    for i in range(days):
        # Add trend component
        trend_component = i * 0.1 if trend else 0

        # Add seasonal components (weekly, monthly, quarterly)
        seasonal_component = 0
        if seasonality:
            # Weekly seasonality (higher on weekends)
            day_of_week = dates[i].weekday()
            weekly = -20 if day_of_week < 5 else 50  # Spend more on weekends

            # Monthly seasonality (higher at beginning and end of month)
            day_of_month = dates[i].day
            days_in_month = (
                dates[i].replace(month=dates[i].month % 12 + 1, day=1)
                - timedelta(days=1)
            ).day
            monthly = 0
            if day_of_month <= 5:
                monthly = 30  # Higher at beginning of month (e.g., rent, bills)
            elif day_of_month >= days_in_month - 5:
                monthly = -40  # Higher spending at end of month

            # Quarterly seasonality (higher at end of quarter)
            month = dates[i].month
            if month in [3, 6, 9, 12] and day_of_month >= 25:
                quarterly = -70  # Higher spending at end of quarter
            else:
                quarterly = 0

            seasonal_component = weekly + monthly + quarterly

        # Add noise
        noise = np.random.normal(
            0, noise_level * abs(base + trend_component + seasonal_component)
        )

        # Calculate daily amount
        amount = base + trend_component + seasonal_component + noise

        # Add some income spikes (e.g. salary) - biweekly
        if i % 14 == 0:
            amount += 2000  # Biweekly salary

        amounts.append(amount)

    # Create DataFrame
    df = pd.DataFrame({"date": dates, "amount": amounts})

    return df


def plot_forecast(historical_df, forecast_df, output_path=None):
    """Plot historical data and forecast"""
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(
        historical_df["date"],
        historical_df["amount"],
        label="Historical Data",
        color="blue",
    )

    # Plot forecast
    plt.plot(forecast_df["date"], forecast_df["amount"], label="Forecast", color="red")

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Amount")
    plt.title("Cash Flow Forecast")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Format x-axis dates
    plt.gcf().autofmt_xdate()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path)

    plt.show()


def test_forecasting_model():
    """Test the cash flow forecasting model"""
    print("\n===== Testing Cash Flow Forecasting Model =====")

    # Generate test data
    print("Generating synthetic test data...")
    df = generate_test_timeseries(days=365, seasonality=True, trend=True)
    print(f"Generated {len(df)} days of test data")

    # Split into training and testing sets
    train_df = df.iloc[:-30]  # Use all but last 30 days for training
    test_df = df.iloc[-30:]  # Use last 30 days for testing

    print(f"Training set: {len(train_df)} days")
    print(f"Testing set: {len(test_df)} days")

    # Train the model
    print("\nTraining the forecasting model...")
    forecaster = CashFlowForecaster(method="ensemble")
    forecaster.fit(train_df, forecast_days=30)

    # Generate forecast
    print("\nGenerating forecast...")
    forecast_df = forecaster.forecast(train_df, forecast_days=30)

    # Evaluate forecast
    print("\nEvaluating forecast accuracy...")
    actual = test_df["amount"].values
    predicted = forecast_df["amount"].values

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot forecast vs actual
    print("\nPlotting forecast vs actual...")
    # Save the plot if a data directory exists
    if os.path.exists("data"):
        plot_forecast(test_df, forecast_df, output_path="data/forecast_test.png")
    else:
        plot_forecast(test_df, forecast_df)

    # Test the service layer
    print("\n===== Testing Forecast Service =====")
    service = ForecastService()

    # Train the service model
    print("Training service model...")
    result = service.train_forecasting_model(
        train_df.to_dict("records"), forecast_days=30
    )
    print(f"Training result: {result['success']}")
    if result["success"]:
        print(f"Model ID: {result['model_id']}")
        print(f"Method: {result['method']}")

    # Test service predictions
    print("\nGenerating service forecast...")
    forecast_result = service.forecast_cash_flow(
        user_id="test_user", transactions=train_df.to_dict("records"), forecast_days=30
    )

    # Print forecast insights
    print("\nForecast Insights:")
    for key, value in forecast_result["insights"].items():
        print(f"  {key}: {value}")

    print("\n===== Forecasting Testing Complete =====")


if __name__ == "__main__":
    test_forecasting_model()

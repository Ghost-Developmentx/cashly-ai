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

    @staticmethod
    def _calculate_confidence(forecast_df):
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

    @staticmethod
    def _simple_forecast(df, current_balance, forecast_days):
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

    def forecast_cash_flow_scenario(
        self, user_id, transactions, forecast_days, adjustments
    ):
        """
        Generate a cash flow forecast with scenario-based adjustments

        Parameters:
        - user_id: User identifier
        - transactions: List of transaction objects with date, amount, category
        - forecast_days: Number of days to forecast
        - adjustments: Dictionary of adjustments to apply to the forecast
            - category_adjustments: Dictionary of category_id -> adjustment_amount
            - income_adjustment: Overall income adjustment amount
            - expense_adjustment: Overall expense adjustment amount
            - recurring_transactions: List of new recurring transactions to add

        Returns:
        - Dictionary with forecast data and insights
        """
        try:
            # First, get the base forecast
            base_forecast = self.forecast_cash_flow(
                user_id, transactions, forecast_days
            )

            # Convert forecast to our structure for scenarios if needed
            if "projected_amount" in base_forecast["forecast"][0]:
                # Convert from projected_amount/projected_balance to income/expenses/balance
                for day in base_forecast["forecast"]:
                    day["income"] = max(0, day.get("projected_amount", 0))
                    day["expenses"] = abs(min(0, day.get("projected_amount", 0)))
                    day["balance"] = day.get("projected_balance", 0)
                    day["net_flow"] = day.get("projected_amount", 0)

            # Apply category-specific adjustments
            if adjustments.get("category_adjustments"):
                category_map = {}
                for transaction in transactions:
                    category = transaction.get("category", "uncategorized")
                    if category not in category_map:
                        category_map[category] = {"income": 0, "expense": 0, "count": 0}

                    amount = transaction.get("amount", 0)
                    if amount >= 0:
                        category_map[category]["income"] += amount
                    else:
                        category_map[category]["expense"] += abs(amount)

                    category_map[category]["count"] += 1

                # Now apply the adjustments to the forecast
                for category_id, adjustment in adjustments[
                    "category_adjustments"
                ].items():
                    if not adjustment or float(adjustment) == 0:
                        continue

                    # Find the category name
                    category_name = None
                    for transaction in transactions:
                        if transaction.get("category_id") == int(category_id):
                            category_name = transaction.get("category")
                            break

                    if not category_name:
                        continue

                    # Distribute the adjustment across the forecast days
                    if category_name in category_map:
                        # If this is an expense category, make the adjustment negative
                        is_expense = (
                            category_map[category_name]["expense"]
                            > category_map[category_name]["income"]
                        )
                        adjustment_amount = float(adjustment)
                        if is_expense:
                            adjustment_amount = -abs(adjustment_amount)

                        # Apply to each day based on historical patterns
                        daily_adjustment = adjustment_amount / forecast_days
                        for day in base_forecast["forecast"]:
                            if is_expense:
                                day["expenses"] += abs(daily_adjustment)
                            else:
                                day["income"] += daily_adjustment

                            day["net_flow"] = day["income"] - day["expenses"]

            # Apply overall income adjustment
            if (
                adjustments.get("income_adjustment")
                and float(adjustments["income_adjustment"]) != 0
            ):
                income_adj = float(adjustments["income_adjustment"]) / forecast_days
                for day in base_forecast["forecast"]:
                    day["income"] += income_adj
                    day["net_flow"] = day["income"] - day["expenses"]

            # Apply overall expense adjustment
            if (
                adjustments.get("expense_adjustment")
                and float(adjustments["expense_adjustment"]) != 0
            ):
                expense_adj = float(adjustments["expense_adjustment"]) / forecast_days
                for day in base_forecast["forecast"]:
                    day["expenses"] += expense_adj
                    day["net_flow"] = day["income"] - day["expenses"]

            # Apply recurring transactions
            if adjustments.get("recurring_transactions"):
                for transaction in adjustments["recurring_transactions"]:
                    if (
                        not transaction.get("amount")
                        or float(transaction["amount"]) == 0
                    ):
                        continue

                    amount = float(transaction["amount"])
                    frequency = transaction.get("frequency", "monthly")

                    # Determine which days to apply the transaction
                    applicable_days = []
                    if frequency == "daily":
                        # Apply to every day
                        applicable_days = list(range(forecast_days))
                    elif frequency == "weekly":
                        # Apply every 7 days
                        applicable_days = list(range(7, forecast_days, 7))
                    elif frequency == "monthly":
                        # Apply on the same day of the month
                        current_day = datetime.now().day
                        for i in range(forecast_days):
                            day_date = datetime.now() + timedelta(days=i)
                            if day_date.day == current_day:
                                applicable_days.append(i)

                    # Apply the transaction to the applicable days
                    for day_index in applicable_days:
                        if day_index >= len(base_forecast["forecast"]):
                            continue

                        if amount >= 0:
                            base_forecast["forecast"][day_index]["income"] += amount
                        else:
                            base_forecast["forecast"][day_index]["expenses"] += abs(
                                amount
                            )

                        base_forecast["forecast"][day_index]["net_flow"] = (
                            base_forecast["forecast"][day_index]["income"]
                            - base_forecast["forecast"][day_index]["expenses"]
                        )

            # Recalculate running balance
            starting_balance = base_forecast.get("insights", {}).get(
                "current_balance", 0
            )
            current_balance = starting_balance
            base_forecast["starting_balance"] = starting_balance

            for day in base_forecast["forecast"]:
                current_balance += day["net_flow"]
                day["balance"] = current_balance

            # Update insights based on the new forecast
            original_insights = base_forecast.get("insights", {})
            scenario_insights = []

            # Compare ending balance with original forecast
            original_ending = original_insights.get(
                "projected_balance", starting_balance
            )
            scenario_ending = base_forecast["forecast"][-1]["balance"]
            difference = scenario_ending - original_ending

            # Add scenario-specific insights
            if abs(difference) > 0:
                direction = "higher" if difference > 0 else "lower"
                scenario_insights.append(
                    {
                        "title": f"Scenario Impact: {direction.capitalize()} Ending Balance",
                        "description": f"This scenario results in a {direction} ending balance of ${abs(difference):.2f} compared to the base forecast.",
                    }
                )

            # Add insights about cash flow stability
            negative_days = sum(
                1 for day in base_forecast["forecast"] if day["balance"] < 0
            )
            if negative_days > 0:
                scenario_insights.append(
                    {
                        "title": "Cash Flow Warning",
                        "description": f"This scenario results in {negative_days} days of negative cash flow. Consider adjusting your plan.",
                    }
                )

            # Add more detailed insights about income and expense changes
            total_income = sum(day["income"] for day in base_forecast["forecast"])
            total_expenses = sum(day["expenses"] for day in base_forecast["forecast"])

            if (
                adjustments.get("income_adjustment")
                and float(adjustments["income_adjustment"]) != 0
            ):
                scenario_insights.append(
                    {
                        "title": "Income Adjustment Impact",
                        "description": f"The income adjustment of ${float(adjustments['income_adjustment']):.2f} changes your monthly income projection.",
                    }
                )

            if (
                adjustments.get("expense_adjustment")
                and float(adjustments["expense_adjustment"]) != 0
            ):
                scenario_insights.append(
                    {
                        "title": "Expense Adjustment Impact",
                        "description": f"The expense adjustment of ${float(adjustments['expense_adjustment']):.2f} changes your monthly expense projection.",
                    }
                )

            # Create return format with both forecast data and insights
            result = {
                "forecast": base_forecast["forecast"],
                "starting_balance": starting_balance,
                "insights": scenario_insights,
            }

            return result

        except Exception as e:
            import traceback

            print(f"Error generating scenario forecast: {str(e)}")
            print(traceback.format_exc())
            return {"error": str(e)}

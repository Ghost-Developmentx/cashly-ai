import pandas as pd
from datetime import datetime
from models.trend_analysis import TrendAnalyzer
from util.model_registry import ModelRegistry


class InsightService:
    """
    Service for generating financial insights and trend analysis
    """

    def __init__(self):
        self.registry = ModelRegistry()
        self.trend_analyzer = TrendAnalyzer(registry=self.registry)

        # Try to load existing model
        try:
            self.trend_analyzer.model, model_info = self.registry.load_model(
                model_type="trend_analysis", latest=True
            )
            self.trend_analyzer.model_id = model_info["id"]
            print(f"Loaded trend analysis model: {model_info['id']}")
        except Exception as e:
            print(f"No existing trend analysis model found: {str(e)}")

    def analyze_trends(self, user_id, transactions, period="3m"):
        """Analyze transaction trends and patterns"""
        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Ensure required columns exist
        required_cols = ["date", "amount"]
        for col in required_cols:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}"}

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Perform trend analysis
        try:
            self.trend_analyzer.analyze(df, period=period)
            analysis_results = self.trend_analyzer.get_analysis()

            # Add user ID to results
            analysis_results["user_id"] = user_id

            # Normalize date formats in the results before returning
            if "forecast" in analysis_results:
                for item in analysis_results["forecast"]:
                    if "date" in item and isinstance(item["date"], (pd.Timestamp, datetime)):
                        item["date"] = item["date"].strftime("%Y-%m-%d")

            if "insights" in analysis_results:
                for insight in analysis_results["insights"]:
                    if "period" in insight and isinstance(insight["period"], (pd.Timestamp, datetime)):
                        insight["period"] = insight["period"].strftime("%Y-%m-%d")

                    # Handle any nested date objects in insight details
                    if "details" in insight and isinstance(insight["details"], dict):
                        for key, value in insight["details"].items():
                            if isinstance(value, (pd.Timestamp, datetime)):
                                insight["details"][key] = value.strftime("%Y-%m-%d")

            # Handle monthly_trends section
            if "monthly_trends" in analysis_results:
                for i, trend in enumerate(analysis_results["monthly_trends"]):
                    if "month" in trend and isinstance(trend["month"], (pd.Timestamp, datetime)):
                        analysis_results["monthly_trends"][i]["month"] = trend["month"].strftime("%Y-%m-%d")

            # Handle day_of_week_spending section (if it exists)
            if "day_of_week_spending" in analysis_results and isinstance(analysis_results["day_of_week_spending"],
                                                                         dict):
                for key, value in list(analysis_results["day_of_week_spending"].items()):
                    if isinstance(key, (pd.Timestamp, datetime)):
                        # Create a new entry with the string date and delete the old one
                        analysis_results["day_of_week_spending"][key.strftime("%Y-%m-%d")] = value
                        del analysis_results["day_of_week_spending"][key]

            # Handle category_breakdown section (if it exists)
            if "category_breakdown" in analysis_results and isinstance(analysis_results["category_breakdown"], dict):
                for category, data in analysis_results["category_breakdown"].items():
                    if isinstance(data, dict):
                        for date_key, amount in list(data.items()):
                            if isinstance(date_key, (pd.Timestamp, datetime)):
                                data[date_key.strftime("%Y-%m-%d")] = amount
                                del data[date_key]

            # Recursively convert any timestamp objects that might be nested deeper
            analysis_results = self._normalize_timestamps_recursive(analysis_results)

            return analysis_results
        except Exception as e:
            print(f"Error analyzing trends: {str(e)}")
            return {"error": str(e)}

    def update_trend_analyzer(self, transactions_data, period=None):
        """
        Update the trend analysis model with new transaction data

        Args:
            transactions_data: List of transaction dictionaries
            period: Analysis period ('1m', '3m', '6m', '1y')

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

        # Update the trend analyzer
        try:
            # The trend analyzer doesn't maintain state in the same way as other models,
            # so we're essentially re-analyzing with the new data
            self.trend_analyzer.update_model(df, period=period)

            return {
                "success": True,
                "model_id": self.trend_analyzer.model_id,
                "message": "Trend analysis updated successfully",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _normalize_timestamps_recursive(self, obj):
        """Recursively convert all timestamp objects to strings in a nested structure"""
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if isinstance(key, (pd.Timestamp, datetime)):
                    # Handle timestamp keys
                    obj[key.strftime("%Y-%m-%d")] = self._normalize_timestamps_recursive(value)
                    del obj[key]
                else:
                    # Process value
                    obj[key] = self._normalize_timestamps_recursive(value)
            return obj
        elif isinstance(obj, list):
            return [self._normalize_timestamps_recursive(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime("%Y-%m-%d")
        else:
            return obj

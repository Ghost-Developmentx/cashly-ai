import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
        """
        Analyze transaction trends and patterns

        Args:
            user_id: User ID
            transactions: List of transaction dictionaries
            period: Analysis period ('1m', '3m', '6m', '1y')

        Returns:
            dict: Trend analysis results
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

        # Perform trend analysis
        try:
            self.trend_analyzer.analyze(df, period=period)
            analysis_results = self.trend_analyzer.get_analysis()

            # Add user ID to results
            analysis_results["user_id"] = user_id

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

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

from app.utils.model_registry import ModelRegistry


class TrendAnalyzer:
    """
    Model for analyzing spending trends, patterns, and providing financial insights
    """

    def __init__(self, registry=None):
        self.model = None
        self.registry = registry or ModelRegistry()
        self.model_id = None

    def analyze(self, transactions_df, period="3m"):
        """
        Analyze transaction data to identify trends and patterns

        Args:
            transactions_df: DataFrame with transaction data
            period: Analysis period ('1m', '3m', '6m', '1y')

        Returns:
            self: The configured model
        """
        # Validate inputs
        if "date" not in transactions_df.columns:
            raise ValueError("Transaction data must include 'date' column")

        if "amount" not in transactions_df.columns:
            raise ValueError("Transaction data must include 'amount' column")

        # Convert period to days
        if period == "1m":
            days = 30
        elif period == "3m":
            days = 90
        elif period == "6m":
            days = 180
        elif period == "1y":
            days = 365
        else:
            # Default to 90 days
            days = 90

        # Process transaction data
        df = transactions_df.copy()

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Filter to specified period
        cutoff_date = datetime.now() - timedelta(days=days)
        period_df = df[df["date"] >= cutoff_date]

        # Calculate transaction metrics
        transaction_count = len(period_df)
        income_sum = period_df[period_df["amount"] > 0]["amount"].sum()
        expense_sum = period_df[period_df["amount"] < 0]["amount"].abs().sum()
        net_flow = income_sum - expense_sum

        # Calculate daily aggregates
        daily = period_df.groupby(period_df["date"].dt.date).agg(
            {"amount": ["sum", "count", "mean"]}
        )
        daily.columns = ["daily_sum", "daily_count", "daily_avg"]

        # Calculate monthly aggregates
        monthly = period_df.groupby(period_df["date"].dt.strftime("%Y-%m")).agg(
            {"amount": ["sum", "count", "mean"]}
        )
        monthly.columns = ["monthly_sum", "monthly_count", "monthly_avg"]

        # Perform category analysis (if category column exists)
        category_insights = []
        if "category" in period_df.columns:
            # Group by category
            category_data = period_df.groupby("category").agg(
                {"amount": ["sum", "count", "mean", "std"]}
            )
            category_data.columns = ["total", "count", "avg", "std"]

            # Calculate growth rates for each category
            category_growth = {}

            # Split period into first and second half
            midpoint = cutoff_date + timedelta(days=days / 2)
            first_half = period_df[
                (period_df["date"] >= cutoff_date) & (period_df["date"] < midpoint)
            ]
            second_half = period_df[period_df["date"] >= midpoint]

            # Calculate spending by category for each half
            for category in period_df["category"].unique():
                first_spend = first_half[first_half["category"] == category][
                    "amount"
                ].sum()
                second_spend = second_half[second_half["category"] == category][
                    "amount"
                ].sum()

                # Calculate growth rate if both periods have data
                if first_spend != 0:
                    growth_rate = (second_spend - first_spend) / abs(first_spend) * 100
                    category_growth[category] = growth_rate

            # Identify recurring transactions
            recurring_candidates = self._identify_recurring_transactions(period_df)

            # Generate category insights
            for category, data in category_data.iterrows():
                if data["total"] == 0:
                    continue

                # Skip income categories (positive amounts)
                if data["total"] > 0:
                    continue

                # Get growth rate
                growth = category_growth.get(category, 0)

                # Calculate volatility (coefficient of variation)
                volatility = data["std"] / abs(data["avg"]) if data["avg"] != 0 else 0

                # Determine spending trend
                if growth < -10:
                    trend = "decreasing"
                elif growth > 10:
                    trend = "increasing"
                else:
                    trend = "stable"

                # Check if category has recurring transactions
                has_recurring = category in [
                    r["category"] for r in recurring_candidates
                ]

                # Generate insight
                insight = {
                    "category": category,
                    "total_spent": abs(data["total"]),
                    "transaction_count": data["count"],
                    "average_transaction": abs(data["avg"]),
                    "spending_trend": trend,
                    "growth_rate": growth,
                    "volatility": volatility,
                    "has_recurring": has_recurring,
                }

                category_insights.append(insight)

        # Generate overall insights
        spending_trend = self._analyze_spending_trend(period_df)
        top_categories = self._get_top_categories(period_df)
        unusual_changes = self._detect_unusual_changes(period_df)
        recurring_items = self._identify_recurring_transactions(period_df)

        # Combine all insights
        analysis_results = {
            "period": period,
            "days_analyzed": days,
            "transaction_count": transaction_count,
            "income_sum": income_sum,
            "expense_sum": expense_sum,
            "net_flow": net_flow,
            "daily_avg_transactions": (
                daily["daily_count"].mean() if not daily.empty else 0
            ),
            "monthly_summary": self._format_monthly_summary(monthly),
            "spending_trend": spending_trend,
            "category_insights": category_insights,
            "top_categories": top_categories,
            "unusual_changes": unusual_changes,
            "recurring_items": recurring_items,
        }

        # Generate actionable insights
        actionable_insights = self._generate_actionable_insights(
            analysis_results, period_df
        )

        analysis_results["insights"] = actionable_insights

        # Store the model
        self.model = analysis_results

        # Save model to registry
        self.model_id = self.registry.save_model(
            model=self.model,
            model_name="trend_analyzer",
            model_type="trend_analysis",
            features=["date", "amount", "category"],
            metrics=None,
            metadata={"period": period, "days": days},
        )

        return self

    def get_analysis(self):
        """
        Get the stored analysis results

        Returns:
            dict: Analysis results
        """
        if self.model is None:
            # Try to load the latest model
            self.model, model_info = self.registry.load_model(
                model_type="trend_analysis", latest=True
            )
            self.model_id = model_info["id"]

        return self.model

    def _format_monthly_summary(self, monthly_data):
        """Format monthly data as a list of dictionaries"""
        if monthly_data.empty:
            return []

        summary = []
        for month, row in monthly_data.iterrows():
            summary.append(
                {
                    "month": month,
                    "total": row["monthly_sum"],
                    "transaction_count": row["monthly_count"],
                    "average_transaction": row["monthly_avg"],
                }
            )

        return summary

    def _analyze_spending_trend(self, transactions_df):
        """Analyze overall spending trend"""
        # Filter to expenses only
        expenses_df = transactions_df[transactions_df["amount"] < 0].copy()
        expenses_df["amount"] = expenses_df["amount"].abs()

        if len(expenses_df) < 14:  # Need at least 2 weeks of data
            return {
                "trend": "insufficient_data",
                "description": "Not enough data to determine trend",
            }

        # Group by day
        expenses_df["date"] = pd.to_datetime(expenses_df["date"])
        daily_expenses = (
            expenses_df.groupby(expenses_df["date"].dt.date)["amount"]
            .sum()
            .reset_index()
        )

        # Sort by date
        daily_expenses = daily_expenses.sort_values("date")

        # Get trend using simple linear regression
        X = np.array(range(len(daily_expenses))).reshape(-1, 1)
        y = daily_expenses["amount"].values

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        r2 = model.score(X, y)

        # Determine trend based on slope and fit quality
        if abs(slope) < 0.1 or r2 < 0.1:
            trend = "stable"
            description = "Your spending pattern has been relatively stable."
        elif slope > 0:
            if slope > 1:
                trend = "rapidly_increasing"
                description = "Your spending is increasing rapidly."
            else:
                trend = "increasing"
                description = "Your spending is gradually increasing."
        else:
            if slope < -1:
                trend = "rapidly_decreasing"
                description = "Your spending is decreasing rapidly."
            else:
                trend = "decreasing"
                description = "Your spending is gradually decreasing."

        return {
            "trend": trend,
            "description": description,
            "slope": float(slope),
            "r2": float(r2),
        }

    def _get_top_categories(self, transactions_df):
        """Get top spending categories"""
        if "category" not in transactions_df.columns:
            return []

        # Filter to expenses only
        expenses_df = transactions_df[transactions_df["amount"] < 0].copy()
        expenses_df["amount"] = expenses_df["amount"].abs()

        # Group by category
        category_spending = (
            expenses_df.groupby("category")["amount"].sum().sort_values(ascending=False)
        )

        # Get top 5 categories
        top_categories = []
        for category, amount in category_spending.head(5).items():
            top_categories.append(
                {
                    "category": category,
                    "amount": float(amount),
                    "percent_of_total": float(
                        amount / expenses_df["amount"].sum() * 100
                    ),
                }
            )

        return top_categories

    def _detect_unusual_changes(self, transactions_df):
        """Detect unusual changes in spending patterns"""
        if "category" not in transactions_df.columns or len(transactions_df) < 30:
            return []

        # Filter to expenses only
        expenses_df = transactions_df[transactions_df["amount"] < 0].copy()
        expenses_df["amount"] = expenses_df["amount"].abs()

        # Group by category and week
        expenses_df["date"] = pd.to_datetime(expenses_df["date"])
        expenses_df["week"] = expenses_df["date"].dt.isocalendar().week

        # Get category spending by week
        weekly_category_spending = (
            expenses_df.groupby(["category", "week"])["amount"].sum().reset_index()
        )

        # Calculate week-over-week changes
        unusual_changes = []

        for category in expenses_df["category"].unique():
            category_data = weekly_category_spending[
                weekly_category_spending["category"] == category
            ]

            if len(category_data) < 2:  # Need at least 2 weeks of data
                continue

            # Calculate week-over-week percentage changes
            category_data = category_data.sort_values("week")
            category_data["prev_amount"] = category_data["amount"].shift(1)
            category_data["pct_change"] = (
                (category_data["amount"] - category_data["prev_amount"])
                / category_data["prev_amount"]
                * 100
            )

            # Identify significant changes (>50% increase or decrease)
            significant_changes = category_data[
                (category_data["pct_change"].abs() > 50)
                & (~category_data["pct_change"].isna())
            ]

            for _, row in significant_changes.iterrows():
                change_type = "increase" if row["pct_change"] > 0 else "decrease"

                unusual_changes.append(
                    {
                        "category": category,
                        "week": int(row["week"]),
                        "previous_amount": float(row["prev_amount"]),
                        "current_amount": float(row["amount"]),
                        "percent_change": float(row["pct_change"]),
                        "change_type": change_type,
                    }
                )

        return unusual_changes

    def _identify_recurring_transactions(self, transactions_df):
        """Identify potentially recurring transactions"""
        if "category" not in transactions_df.columns or len(transactions_df) < 30:
            return []

        # Convert date to datetime
        df = transactions_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Group by category and month
        df["month"] = df["date"].dt.strftime("%Y-%m")
        monthly_counts = (
            df.groupby(["category", "month"]).size().reset_index(name="count")
        )

        # Calculate consistency of monthly transactions
        category_consistency = monthly_counts.groupby("category").agg(
            {"count": ["mean", "std", "count"]}
        )
        category_consistency.columns = [
            "avg_transactions",
            "std_transactions",
            "months_present",
        ]

        # Calculate coefficient of variation (lower means more consistent)
        category_consistency["consistency"] = (
            category_consistency["std_transactions"]
            / category_consistency["avg_transactions"]
        )

        # Filter for categories that appear in most months with consistent transaction counts
        recurring_candidates = []

        for category, row in category_consistency.iterrows():
            # Skip income categories
            category_amounts = transactions_df[transactions_df["category"] == category][
                "amount"
            ]
            if category_amounts.mean() > 0:
                continue

            # Check if category appears consistently
            total_months = df["month"].nunique()
            if row["months_present"] / total_months < 0.7:
                continue

            # Check for consistent transaction count
            if row["consistency"] > 0.5:
                continue

            # Get typical transaction amount
            category_df = transactions_df[transactions_df["category"] == category]
            typical_amount = category_df["amount"].median()

            recurring_candidates.append(
                {
                    "category": category,
                    "frequency": (
                        "monthly" if row["avg_transactions"] >= 0.9 else "occasional"
                    ),
                    "typical_amount": float(typical_amount),
                    "consistency": float(row["consistency"]),
                    "confidence": float(
                        min(
                            0.9,
                            (1 - row["consistency"])
                            * (row["months_present"] / total_months),
                        )
                    ),
                }
            )

        return recurring_candidates

    def _generate_actionable_insights(self, analysis_results, transactions_df):
        """Generate actionable insights based on the analysis"""
        insights = []

        # Spending Trend Insight
        if "spending_trend" in analysis_results:
            trend = analysis_results["spending_trend"]["trend"]

            if trend == "rapidly_increasing":
                insights.append(
                    {
                        "type": "warning",
                        "title": "Spending Increasing Rapidly",
                        "description": "Your overall spending has been increasing significantly. Review your recent expenses to identify areas where you can cut back.",
                    }
                )
            elif trend == "increasing":
                insights.append(
                    {
                        "type": "info",
                        "title": "Spending Trending Upward",
                        "description": "Your spending is gradually increasing. Keep an eye on your expenses to ensure they stay within your budget.",
                    }
                )
            elif trend == "decreasing":
                insights.append(
                    {
                        "type": "success",
                        "title": "Reduced Spending",
                        "description": "You've been reducing your spending. Great job managing your finances!",
                    }
                )

        # Top Spending Categories Insight
        if analysis_results["top_categories"]:
            top_category = analysis_results["top_categories"][0]

            insights.append(
                {
                    "type": "info",
                    "title": f"Largest Expense: {top_category['category'].title()}",
                    "description": f"Your largest expense category is {top_category['category'].title()}, accounting for {top_category['percent_of_total']:.1f}% of your spending.",
                }
            )

        # Unusual Changes Insights
        for change in analysis_results["unusual_changes"][:2]:  # Limit to 2 insights
            if change["change_type"] == "increase":
                insights.append(
                    {
                        "type": "warning",
                        "title": f"Unusual Increase in {change['category'].title()}",
                        "description": f"Your spending on {change['category'].title()} increased by {abs(change['percent_change']):.1f}% recently.",
                    }
                )
            else:
                insights.append(
                    {
                        "type": "success",
                        "title": f"Significant Decrease in {change['category'].title()}",
                        "description": f"Your spending on {change['category'].title()} decreased by {abs(change['percent_change']):.1f}% recently.",
                    }
                )

        # Recurring Expenses Insight
        if analysis_results["recurring_items"]:
            total_recurring = sum(
                abs(item["typical_amount"])
                for item in analysis_results["recurring_items"]
            )

            insights.append(
                {
                    "type": "info",
                    "title": "Recurring Expenses",
                    "description": f"You have approximately ${total_recurring:.2f} in recurring monthly expenses across {len(analysis_results['recurring_items'])} categories.",
                }
            )

        # Cash Flow Insight
        if analysis_results["net_flow"] < 0:
            insights.append(
                {
                    "type": "alert",
                    "title": "Negative Cash Flow",
                    "description": f"Your expenses exceed your income by ${abs(analysis_results['net_flow']):.2f} during this period. Consider ways to increase income or reduce expenses.",
                }
            )
        else:
            # Only show positive cash flow insight if it's significant
            if analysis_results["net_flow"] > 0.1 * analysis_results["income_sum"]:
                insights.append(
                    {
                        "type": "success",
                        "title": "Positive Cash Flow",
                        "description": f"You're saving ${analysis_results['net_flow']:.2f} during this period, which is {(analysis_results['net_flow'] / analysis_results['income_sum'] * 100):.1f}% of your income. Consider investing this surplus.",
                    }
                )

        # Category-specific insights
        for category_insight in analysis_results["category_insights"]:
            # High volatility insight
            if category_insight["volatility"] > 1.0:
                insights.append(
                    {
                        "type": "info",
                        "title": f"Inconsistent {category_insight['category'].title()} Spending",
                        "description": f"Your spending on {category_insight['category'].title()} varies significantly from month to month. Consider setting a consistent budget for this category.",
                    }
                )

            # Rapid growth insight
            if category_insight["growth_rate"] > 50:
                insights.append(
                    {
                        "type": "warning",
                        "title": f"Rapidly Increasing {category_insight['category'].title()} Expenses",
                        "description": f"Your spending on {category_insight['category'].title()} has increased by {category_insight['growth_rate']:.1f}% recently. Review these expenses to ensure they're necessary.",
                    }
                )

        return insights

    def update_model(self, new_transactions_df, period=None):
        """
        Update the trend analysis model with new transaction data

        Args:
            new_transactions_df: DataFrame with new transaction data
            period: Analysis period ('1m', '3m', '6m', '1y')

        Returns:
            self: The updated model
        """
        if self.model is None:
            # If no model exists, just train from scratch
            return self.analyze(new_transactions_df, period=period or "3m")

        # Store previous model ID
        previous_model_id = self.model_id

        # If period is specified, use it; otherwise keep the current period
        current_period = self.model.get("period", "3m")
        analysis_period = period or current_period

        # Analyze the new data
        analysis_results = self.analyze(
            new_transactions_df, period=analysis_period
        ).get_analysis()

        # No need to update further as trend analysis is primarily analytical
        # and doesn't build a persistent model that learns

        # Update metadata to record the update history
        self.model_id = self.registry.save_model(
            model=analysis_results,
            model_name="trend_analyzer_updated",
            model_type="trend_analysis",
            features=["date", "amount", "category"],
            metrics=None,
            metadata={
                "period": analysis_period,
                "days": analysis_results.get("days_analyzed", 90),
                "previous_model_id": previous_model_id,
                "update_time": datetime.now().isoformat(),
            },
        )

        return self

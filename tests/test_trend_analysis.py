import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import sys
import matplotlib.pyplot as plt
from models.trend_analysis import TrendAnalyzer
from services.insight_service import InsightService


def generate_trend_data(
    n_days=180, categories=None, trend_factor=0.1, seasonality=True
):
    """Generate synthetic transaction data with trends and patterns"""

    if categories is None:
        categories = {
            "groceries": {"type": "essential", "weekly_base": 150, "trend": 0.05},
            "dining": {"type": "discretionary", "weekly_base": 100, "trend": 0.15},
            "utilities": {"type": "essential", "weekly_base": 50, "trend": 0.0},
            "rent": {"type": "essential", "weekly_base": 500, "trend": 0.0},
            "transportation": {"type": "essential", "weekly_base": 60, "trend": -0.05},
            "entertainment": {"type": "discretionary", "weekly_base": 80, "trend": 0.1},
            "shopping": {"type": "discretionary", "weekly_base": 120, "trend": 0.2},
            "healthcare": {"type": "essential", "weekly_base": 40, "trend": 0.0},
            "subscription": {"type": "discretionary", "weekly_base": 30, "trend": 0.05},
            "income": {"type": "income", "weekly_base": 1500, "trend": 0.02},
        }

    # Generate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Generate transactions
    transactions = []

    for category, props in categories.items():
        category_type = props["type"]
        weekly_base = props["weekly_base"]
        trend = props["trend"]

        # Determine transaction frequency
        if category == "rent":
            frequency = "monthly"
        elif category == "utilities":
            frequency = "monthly"
        elif category == "income":
            frequency = "biweekly"
        elif category == "subscription":
            frequency = "monthly"
        else:
            frequency = "weekly"

        # Generate transactions based on frequency
        for i, date in enumerate(dates):
            # Apply trend factor
            trend_multiplier = 1 + (trend * trend_factor * i / n_days)

            # Apply seasonality
            seasonality_factor = 1.0
            if seasonality:
                # Weekly seasonality (weekends vs weekdays)
                if date.weekday() >= 5:  # Weekend
                    if category in ["dining", "entertainment", "shopping"]:
                        seasonality_factor *= 1.5
                    elif category in ["groceries"]:
                        seasonality_factor *= 1.2

                # Monthly seasonality (beginning/end of month)
                day_of_month = date.day
                if day_of_month <= 5:  # Beginning of month
                    if category in ["rent", "utilities"]:
                        seasonality_factor *= 1.0  # Regular bills
                elif day_of_month >= 25:  # End of month
                    if category in ["shopping"]:
                        seasonality_factor *= 1.3  # End of month shopping

                # Quarterly seasonality
                month = date.month
                if month in [3, 6, 9, 12] and day_of_month >= 25:  # End of quarter
                    if category in ["entertainment", "shopping"]:
                        seasonality_factor *= 1.4  # End of quarter splurge

            # Determine if we create a transaction on this date
            create_transaction = False

            if frequency == "daily":
                create_transaction = True
            elif frequency == "weekly":
                # Once per week, roughly
                create_transaction = date.weekday() == random.randint(0, 6)
            elif frequency == "biweekly":
                # Twice per month, roughly
                create_transaction = date.day in [1, 15] or (
                    date.day == 2 and random.random() < 0.5
                )
            elif frequency == "monthly":
                # Once per month, at beginning
                create_transaction = date.day == 1 or (
                    date.day == 2 and random.random() < 0.5
                )

            # Add some randomness to transaction creation
            if random.random() < 0.2:  # 20% chance to flip decision
                create_transaction = not create_transaction

            if create_transaction:
                # Calculate amount
                base_amount = weekly_base

                if frequency == "monthly":
                    base_amount = weekly_base * 4  # Monthly amount is 4x weekly

                # Apply trend and seasonality
                adjusted_amount = base_amount * trend_multiplier * seasonality_factor

                # Add noise
                noise_factor = random.uniform(0.8, 1.2)
                final_amount = adjusted_amount * noise_factor

                # Set sign based on category type
                if category_type == "income":
                    amount = abs(final_amount)  # Income is positive
                else:
                    amount = -abs(final_amount)  # Expenses are negative

                # Threshold to avoid tiny transactions
                if abs(amount) >= 1.0:
                    transactions.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "amount": round(amount, 2),
                            "category": category,
                            "description": f"{category.title()} {'Income' if amount > 0 else 'Payment'}",
                        }
                    )

    # Add some random one-off transactions
    n_random = int(n_days * 0.1)  # 10% of days get random transactions
    random_categories = ["shopping", "healthcare", "entertainment", "dining"]

    for _ in range(n_random):
        date = random.choice(dates)
        category = random.choice(random_categories)
        amount = -random.uniform(20, 200)  # Random amount

        transactions.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "amount": round(amount, 2),
                "category": category,
                "description": f"Random {category.title()} Expense",
            }
        )

    # Sort by date
    transactions = sorted(transactions, key=lambda x: x["date"])

    return pd.DataFrame(transactions)


def plot_spending_trends(df, categories=None):
    """Plot spending trends over time"""
    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Filter to expenses only (negative amounts)
    expenses_df = df[df["amount"] < 0].copy()
    expenses_df["amount"] = expenses_df["amount"].abs()

    # Resample by week
    weekly_expenses = (
        expenses_df.groupby([pd.Grouper(key="date", freq="W"), "category"])["amount"]
        .sum()
        .reset_index()
    )

    # Select top categories if not specified
    if not categories:
        top_categories = (
            expenses_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
    else:
        top_categories = categories

    # Plot
    plt.figure(figsize=(12, 6))

    for category in top_categories:
        category_data = weekly_expenses[weekly_expenses["category"] == category]
        plt.plot(
            category_data["date"],
            category_data["amount"],
            marker="o",
            linestyle="-",
            label=category,
        )

    plt.xlabel("Date")
    plt.ylabel("Weekly Spending ($)")
    plt.title("Weekly Spending Trends by Category")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Save plot if data directory exists
    if os.path.exists("../data"):
        plt.savefig("data/spending_trends.png")

    plt.tight_layout()
    plt.show()


def test_trend_analysis():
    """Test the trend analysis model"""
    print("\n===== Testing Trend Analysis Model =====")

    # Generate test data
    print("Generating synthetic test data...")
    df = generate_trend_data(n_days=180)
    print(
        f"Generated {len(df)} test transactions across {df['category'].nunique()} categories"
    )

    # Plot spending trends
    print("\nPlotting spending trends...")
    plot_spending_trends(
        df, categories=["groceries", "dining", "shopping", "entertainment"]
    )

    # Analyze trends
    print("\nAnalyzing trends...")
    analyzer = TrendAnalyzer()
    analyzer.analyze(df, period="6m")

    # Get analysis results
    results = analyzer.get_analysis()

    # Print overall metrics
    print("\nOverall Metrics:")
    print(f"Period: {results['period']} ({results['days_analyzed']} days)")
    print(f"Transaction Count: {results['transaction_count']}")
    print(f"Total Income: ${results['income_sum']:.2f}")
    print(f"Total Expenses: ${results['expense_sum']:.2f}")
    print(f"Net Cash Flow: ${results['net_flow']:.2f}")

    # Print spending trend
    print("\nSpending Trend:")
    print(f"Trend: {results['spending_trend']['trend']}")
    print(f"Description: {results['spending_trend']['description']}")

    # Print top categories
    print("\nTop Spending Categories:")
    for i, category in enumerate(results["top_categories"]):
        print(
            f"{i + 1}. {category['category']}: ${category['amount']:.2f} ({category['percent_of_total']:.1f}%)"
        )

    # Print recurring items
    print("\nRecurring Transactions:")
    for item in results["recurring_items"]:
        print(
            f"{item['category']}: ${abs(item['typical_amount']):.2f} ({item['frequency']}, {item['confidence'] * 100:.1f}% confidence)"
        )

    # Print unusual changes
    if results["unusual_changes"]:
        print("\nUnusual Spending Changes:")
        for change in results["unusual_changes"]:
            print(
                f"{change['category']}: {change['change_type']} of {abs(change['percent_change']):.1f}% (${change['previous_amount']:.2f} → ${change['current_amount']:.2f})"
            )

    # Print insights
    print("\nActionable Insights:")
    for insight in results["insights"]:
        print(f"[{insight['type']}] {insight['title']}: {insight['description']}")

    # Test the service layer
    print("\n===== Testing Insight Service =====")
    service = InsightService()

    # Generate service insights
    print("Generating insights via service...")
    insights = service.analyze_trends(
        user_id="test_user", transactions=df.to_dict("records"), period="3m"
    )

    # Print service insights
    print("\nService Insights:")
    for insight in insights["insights"]:
        print(f"[{insight['type']}] {insight['title']}: {insight['description']}")

    print("\n===== Trend Analysis Testing Complete =====")


if __name__ == "__main__":
    test_trend_analysis()

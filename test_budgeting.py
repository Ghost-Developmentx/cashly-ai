import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import sys
from models.budgeting import BudgetRecommender
from services.budget_service import BudgetService


def generate_test_budget_data(n_samples=1000, n_months=3):
    """Generate synthetic transaction data for testing budget recommendations"""

    # Define categories and their properties
    categories = {
        # Essentials
        "housing": {
            "type": "essential",
            "min": 500,
            "max": 2000,
            "frequency": "monthly",
        },
        "utilities": {
            "type": "essential",
            "min": 100,
            "max": 300,
            "frequency": "monthly",
        },
        "groceries": {
            "type": "essential",
            "min": 50,
            "max": 200,
            "frequency": "weekly",
        },
        "transportation": {
            "type": "essential",
            "min": 30,
            "max": 150,
            "frequency": "weekly",
        },
        "healthcare": {
            "type": "essential",
            "min": 20,
            "max": 300,
            "frequency": "monthly",
        },
        # Discretionary
        "dining": {
            "type": "discretionary",
            "min": 10,
            "max": 100,
            "frequency": "weekly",
        },
        "entertainment": {
            "type": "discretionary",
            "min": 20,
            "max": 150,
            "frequency": "weekly",
        },
        "shopping": {
            "type": "discretionary",
            "min": 20,
            "max": 200,
            "frequency": "biweekly",
        },
        "subscription": {
            "type": "discretionary",
            "min": 5,
            "max": 50,
            "frequency": "monthly",
        },
        "travel": {
            "type": "discretionary",
            "min": 100,
            "max": 1000,
            "frequency": "quarterly",
        },
        # Savings
        "savings": {"type": "savings", "min": 100, "max": 500, "frequency": "monthly"},
        "investment": {
            "type": "savings",
            "min": 50,
            "max": 1000,
            "frequency": "monthly",
        },
        "debt_payment": {
            "type": "savings",
            "min": 100,
            "max": 800,
            "frequency": "monthly",
        },
    }

    # Generate random transactions
    transactions = []

    # Get the current date
    current_date = datetime.now()

    # Start date is n_months ago
    start_date = current_date - timedelta(days=30 * n_months)

    # Generate transactions for each month
    for month in range(n_months):
        month_start = start_date + timedelta(days=30 * month)
        month_end = month_start + timedelta(days=30)

        # Generate transactions for each category
        for category, props in categories.items():
            category_type = props["type"]
            min_amount = props["min"]
            max_amount = props["max"]
            frequency = props["frequency"]

            # Determine how many transactions per month based on frequency
            if frequency == "monthly":
                num_transactions = 1
            elif frequency == "biweekly":
                num_transactions = 2
            elif frequency == "weekly":
                num_transactions = 4
            elif frequency == "quarterly":
                # Only add quarterly transactions in the first and last month
                if month % 3 == 0:
                    num_transactions = 1
                else:
                    num_transactions = 0
            else:
                num_transactions = 1

            # Add randomness to the number of transactions
            num_transactions = max(
                0, round(num_transactions * random.uniform(0.8, 1.2))
            )

            # Generate transactions for this category and month
            for _ in range(num_transactions):
                # Random date within the month
                days_offset = random.randint(0, 29)
                transaction_date = month_start + timedelta(days=days_offset)

                # Random amount
                amount = -round(random.uniform(min_amount, max_amount), 2)

                # Add transaction
                transactions.append(
                    {
                        "date": transaction_date.strftime("%Y-%m-%d"),
                        "description": f"{category.title()} Payment",
                        "amount": amount,
                        "category": category,
                    }
                )

    # Add income transactions
    monthly_income = 5000
    for month in range(n_months):
        month_start = start_date + timedelta(days=30 * month)

        # Biweekly income
        for i in range(2):
            transaction_date = month_start + timedelta(days=i * 14)

            # Add transaction
            transactions.append(
                {
                    "date": transaction_date.strftime("%Y-%m-%d"),
                    "description": "Salary Deposit",
                    "amount": monthly_income / 2,  # Half of monthly income each payment
                    "category": "income",
                }
            )

    return pd.DataFrame(transactions), monthly_income


def test_budget_recommender():
    """Test the budget recommendation model"""
    print("\n===== Testing Budget Recommendation Model =====")

    # Generate test data
    print("Generating synthetic test data...")
    df, monthly_income = generate_test_budget_data(n_months=3)
    print(
        f"Generated {len(df)} test transactions across {df['category'].nunique()} categories"
    )
    print(f"Monthly income: ${monthly_income}")

    # Train the model
    print("\nTraining the budget recommendation model...")
    recommender = BudgetRecommender()
    recommender.fit(df, monthly_income=monthly_income)

    # Get category clusters
    print("\nCategory Clusters:")
    for category, budget_type in recommender.category_clusters.items():
        print(f"  {category}: {budget_type}")

    # Generate budget recommendations
    print("\nGenerating budget recommendations...")
    recommendations = recommender.recommend_budget(monthly_income=monthly_income)

    # Print recommendations
    print("\nBudget Recommendations:")
    total_recommended = 0

    # Group by type
    recommendations_by_type = {"essentials": {}, "discretionary": {}, "savings": {}}

    for category, amount in recommendations.items():
        budget_type = recommender.category_clusters.get(
            category,
            recommender._categorize_new_category(
                category, recommender.category_mappings
            ),
        )
        recommendations_by_type[budget_type][category] = amount
        total_recommended += amount

    # Print by type
    for budget_type, cats in recommendations_by_type.items():
        type_total = sum(cats.values())
        type_percent = (type_total / monthly_income) * 100

        print(f"\n{budget_type.title()} (${type_total:.2f}, {type_percent:.1f}%):")

        for category, amount in cats.items():
            percent = (amount / monthly_income) * 100
            print(f"  {category}: ${amount:.2f} ({percent:.1f}%)")

    print(
        f"\nTotal Recommended: ${total_recommended:.2f} ({(total_recommended / monthly_income) * 100:.1f}% of income)"
    )

    # Test the service layer
    print("\n===== Testing Budget Service =====")
    service = BudgetService()

    # Train the service model
    print("Training service model...")
    result = service.train_budget_model(
        df.to_dict("records"), monthly_income=monthly_income
    )
    print(f"Training result: {result['success']}")
    if result["success"]:
        print(f"Model ID: {result['model_id']}")

    # Test service recommendations
    print("\nGenerating service budget recommendations...")
    budget_result = service.generate_budget(
        user_id="test_user",
        transactions=df.to_dict("records"),
        monthly_income=monthly_income,
    )

    # Print recommendation summary
    print("\nService Recommendations Summary:")
    print(f"Monthly Income: ${budget_result['monthly_income']}")
    print(f"Total Budget: ${budget_result['recommended_budget']['total']}")

    # Print by type
    for budget_type, data in budget_result["recommended_budget"]["by_type"].items():
        type_total = data["total"]
        type_percent = (type_total / monthly_income) * 100

        print(f"\n{budget_type.title()} (${type_total:.2f}, {type_percent:.1f}%):")

        for category, amount in data["categories"].items():
            percent = (amount / monthly_income) * 100
            print(f"  {category}: ${amount:.2f} ({percent:.1f}%)")

    # Print insights
    print("\nBudgeting Insights:")

    if "rule_comparison" in budget_result["insights"]:
        print("50/30/20 Rule Comparison:")
        current = budget_result["insights"]["rule_comparison"]["current_allocation"]
        ideal = budget_result["insights"]["rule_comparison"]["ideal_allocation"]

        print(
            f"  Current: Essentials {current['essentials']}%, Discretionary {current['discretionary']}%, Savings {current['savings']}%"
        )
        print(
            f"  Ideal:   Essentials {ideal['essentials']}%, Discretionary {ideal['discretionary']}%, Savings {ideal['savings']}%"
        )

    if "top_spending_categories" in budget_result["insights"]:
        print("\nTop Spending Categories:")
        for item in budget_result["insights"]["top_spending_categories"]:
            print(f"  {item['category']}: ${item['amount']:.2f}")

    print("\n===== Budget Testing Complete =====")


if __name__ == "__main__":
    test_budget_recommender()

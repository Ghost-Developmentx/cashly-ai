from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from flask_cors import CORS
import datetime

app = Flask(__name__)

CORS(app)

# Directory For Saving Trained Models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check to verify service is running"""
    return jsonify({"status": "healthy", "service": "cashly-ai-service"}), 200


@app.route("/categorize/transaction", methods=["POST"])
def categorize_transaction():
    """
    Categorize a transaction based on its description

    Expected input:
    {
        "description": "AMAZON PAYMENT",
        "amount": -45.67,
        "date": "2025-03-10"
    }
    -------
    """

    try:
        data = request.json
        description = data.get("description", "").lower()
        amount = data.get("amount", 0)

        # Simple rule-based categorization
        categories = {
            "grocery": ["grocery", "supermarket", "food", "walmart", "target"],
            "utilities": ["electric", "water", "gas", "utility", "utilities"],
            "subscription": ["netflix", "spotify", "subscription", "monthly"],
            "shopping": ["amazon", "ebay", "purchase", "buy"],
            "dining": ["restaurant", "cafe", "coffee", "dining"],
            "income": ["salary", "deposit", "income", "payment received"],
            "transportation": ["uber", "lyft", "taxi", "train", "transit"],
            "housing": ["rent", "mortgage", "housing"],
            "healthcare": ["doctor", "pharmacy", "medical", "health"],
            "entertainment": ["movie", "ticket", "entertainment"],
        }

        # Simple amount-based categorization
        if amount > 0:
            return jsonify({"category": "income", "confidence": 0.9}), 200

        # Text-based categorization
        best_category = "uncategorized"
        best_score = 0

        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in description)
            if score > best_score:
                best_score = score
                best_category = category

        confidence = min(0.5 + (best_score * 0.1), 0.95) if best_score > 0 else 0.3

        return (
            jsonify(
                {
                    "category": best_category,
                    "confidence": confidence,
                    "alternative_categories": [
                        c for c in categories.keys() if c != best_category
                    ][:3],
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast/cash_flow", methods=["POST"])
def forecast_cash_flow():
    """
    Forecast future cash flow based on historical cash flow

    Expected input:
    {
        "user_id": "user_123",
        "transactions": [
            {"date": "2025-01-01", "amount": 1000.00, "category": "income"},
            {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
            ...
        ],
        "forecast_days": 30
    }
    -------
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        transactions = data.get("transactions", [])
        forecast_days = data.get("forecast_days", 30)

        if not user_id or not transactions:
            return (
                jsonify({"status": "failed", "message": "Missing required parameter"}),
                400,
            )

        # Convert To Dataframe
        df = pd.DataFrame(transactions)
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month

        daily_net = df.groupby(df["date"].dt.date)["amount"].sum()

        # Simple Forecasting Technique for MVP
        avg_daily_net = daily_net.mean()
        std_dev = daily_net.std() if len(daily_net) > 1 else avg_daily_net * 0.1

        # Generate forecast with some randomness to simulate fluctuations
        last_date = df["date"].max()
        forecast_dates = [
            last_date + datetime.timedelta(days=i + 1) for i in range(forecast_days)
        ]
        forecast_values = [
            avg_daily_net + np.random.normal(0, std_dev) for _ in range(forecast_days)
        ]

        current_balance = df["amount"].sum()
        cumulative_forecast = [current_balance]
        running_total = current_balance

        for value in forecast_values:
            running_total += value
            cumulative_forecast.append(running_total)

        forecast_dates = [last_date] + forecast_dates

        # Response

        response = {
            "forecast": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "projected_balance": round(balance, 2),
                }
                for date, balance in zip(forecast_dates, cumulative_forecast)
            ],
            "insights": {
                "average_daily_net": round(avg_daily_net, 2),
                "current_balance": round(current_balance, 2),
                "projected_30_day_balance": round(cumulative_forecast[-1], 2),
                "cash_flow_trend": "positive" if avg_daily_net > 0 else "negative",
            },
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate/budget", methods=["POST"])
def generate_budget():
    """
    Generate budget recommendations based on spending patterns

    Expected input:
    {
        "user_id": "user_123",
        "transactions": [
            {"date": "2025-01-01", "amount": -50.00, "category": "groceries"},
            {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
            ...
            ]
        "income": 5000.00
    }
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        transactions = data.get("transactions", [])
        monthly_income = data.get("income", 0)

        if not user_id or not transactions:
            return jsonify({"error": "Missing required data"}), 400

        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        df["date"] = pd.to_datetime(df["date"])
        df["amount"] = df["amount"].astype(float)

        # Filter only expenses (negative amounts)
        expenses = df[df["amount"] < 0].copy()
        expenses["amount"] = expenses["amount"].abs()

        # Aggregate by category
        category_spending = expenses.groupby("category")["amount"].sum().to_dict()
        total_spending = expenses["amount"].sum()

        # Calculate percentage of income
        category_percentages = {
            cat: round((amt / monthly_income) * 100, 1)
            for cat, amt in category_spending.items()
        }

        # Generate budget recommendations based on 50/30/20 rule
        # 50% needs, 30% wants, 20% savings

        # Simple categorization of expense types
        needs = ["housing", "utilities", "groceries", "healthcare", "transportation"]
        wants = ["dining", "entertainment", "shopping", "subscription"]
        savings = ["savings", "investment"]

        # Calculate current allocation
        current_needs = sum(category_spending.get(cat, 0) for cat in needs)
        current_wants = sum(category_spending.get(cat, 0) for cat in wants)
        current_savings = sum(category_spending.get(cat, 0) for cat in savings)

        # Calculate ideal allocation
        ideal_needs = monthly_income * 0.5
        ideal_wants = monthly_income * 0.3
        ideal_savings = monthly_income * 0.2

        # Calculate adjustment factors
        needs_adjustment = (ideal_needs / current_needs) if current_needs > 0 else 1
        wants_adjustment = (ideal_wants / current_wants) if current_wants > 0 else 1

        # Generate recommended budget
        budget_recommendations = {}

        for category, amount in category_spending.items():
            if category in needs:
                adjustment = needs_adjustment
            elif category in wants:
                adjustment = wants_adjustment
            else:
                adjustment = 1  # No adjustment for other categories

            recommended = round(
                min(amount * adjustment, amount * 1.2)
            )  # Cap at 20% increase
            budget_recommendations[category] = recommended

        # Add savings if not present
        if "savings" not in budget_recommendations:
            budget_recommendations["savings"] = round(ideal_savings)

        # Calculate surplus/deficit
        total_budget = sum(budget_recommendations.values())
        surplus_deficit = monthly_income - total_budget

        response = {
            "monthly_income": monthly_income,
            "current_spending": {
                "total": round(total_spending, 2),
                "by_category": {k: round(v, 2) for k, v in category_spending.items()},
                "percentages": category_percentages,
            },
            "recommended_budget": {
                "total": round(total_budget, 2),
                "by_category": {
                    k: round(v, 2) for k, v in budget_recommendations.items()
                },
                "surplus_deficit": round(surplus_deficit, 2),
            },
            "insights": {
                "current_allocation": {
                    "needs": round(
                        (
                            (current_needs / monthly_income) * 100
                            if monthly_income > 0
                            else 0
                        ),
                        1,
                    ),
                    "wants": round(
                        (
                            (current_wants / monthly_income) * 100
                            if monthly_income > 0
                            else 0
                        ),
                        1,
                    ),
                    "savings": round(
                        (
                            (current_savings / monthly_income) * 100
                            if monthly_income > 0
                            else 0
                        ),
                        1,
                    ),
                },
                "ideal_allocation": {"needs": 50, "wants": 30, "savings": 20},
                "highest_expense_categories": sorted(
                    category_spending.items(), key=lambda x: x[1], reverse=True
                )[:3],
            },
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/trends", methods=["POST"])
def analyze_trends():
    """
    Analyze financial trends and patterns

    Expected input:
    {
        "user_id": "user_123",
        "transactions": [
            {"date": "2025-01-01", "amount": -50.00, "category": "groceries"},
            {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
            ...
        ],
        "period": "6m"  # 1m, 3m, 6m, 1y
    }
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        transactions = data.get("transactions", [])
        _period = data.get("period", "3m")

        if not user_id or not transactions:
            return jsonify({"error": "Missing required data"}), 400

        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        df["date"] = pd.to_datetime(df["date"])
        df["amount"] = df["amount"].astype(float)
        df["month"] = df["date"].dt.strftime("%Y-%m")

        # Monthly aggregation
        monthly_totals = df.groupby("month")["amount"].sum()
        monthly_expenses = df[df["amount"] < 0].groupby("month")["amount"].sum().abs()
        monthly_income = df[df["amount"] > 0].groupby("month")["amount"].sum()

        # Category aggregation
        category_monthly = (
            df.groupby(["month", "category"])["amount"].sum().reset_index()
        )

        # Find recurring expenses
        # For simplicity, we'll consider transactions with the same description and similar amounts
        df["month_day"] = df["date"].dt.day
        recurring = []

        for category in df["category"].unique():
            cat_df = df[df["category"] == category]
            monthly_count = cat_df.groupby("month").size()
            # If appears in at least 2 months and amount is similar
            if len(monthly_count) >= 2 and monthly_count.mean() >= 1:
                recurring.append(
                    {
                        "category": category,
                        "average_amount": round(cat_df["amount"].mean(), 2),
                        "frequency": (
                            "monthly" if monthly_count.mean() >= 0.8 else "occasional"
                        ),
                        "confidence": min(0.5 + (monthly_count.mean() * 0.1), 0.95),
                    }
                )

        # Identify trends (simple implementation)
        income_trend = "stable"
        if len(monthly_income) >= 2:
            last_month = monthly_income.iloc[-1]
            prev_month = monthly_income.iloc[-2]
            pct_change = ((last_month - prev_month) / prev_month) * 100

            if pct_change > 5:
                income_trend = "increasing"
            elif pct_change < -5:
                income_trend = "decreasing"

        expense_trend = "stable"
        if len(monthly_expenses) >= 2:
            last_month = monthly_expenses.iloc[-1]
            prev_month = monthly_expenses.iloc[-2]
            pct_change = ((last_month - prev_month) / prev_month) * 100

            if pct_change > 5:
                expense_trend = "increasing"
            elif pct_change < -5:
                expense_trend = "decreasing"

        # Generate insights
        insights = []

        # Income vs Expenses
        savings_rate = []
        for month in set(monthly_income.index).intersection(
            set(monthly_expenses.index)
        ):
            income = monthly_income.get(month, 0)
            expense = monthly_expenses.get(month, 0)
            if income > 0:
                savings_rate.append((income - expense) / income)

        avg_savings_rate = sum(savings_rate) / len(savings_rate) if savings_rate else 0

        if avg_savings_rate < 0.1:
            insights.append(
                {
                    "type": "warning",
                    "title": "Low Savings Rate",
                    "description": "Your average savings rate is below 10%. Consider reducing non-essential expenses.",
                }
            )

        # Unusual spending
        if len(monthly_expenses) >= 3:
            avg_expenses = monthly_expenses.iloc[:-1].mean()
            last_month_expenses = monthly_expenses.iloc[-1]

            if last_month_expenses > avg_expenses * 1.2:
                insights.append(
                    {
                        "type": "alert",
                        "title": "Unusual Spending",
                        "description": f"Your expenses last month were {round((last_month_expenses / avg_expenses - 1) * 100, 1)}% higher than average.",
                    }
                )

        # Format response
        response = {
            "monthly_summary": [
                {
                    "month": month,
                    "income": round(monthly_income.get(month, 0), 2),
                    "expenses": round(monthly_expenses.get(month, 0), 2),
                    "net": round(monthly_totals.get(month, 0), 2),
                }
                for month in sorted(set(monthly_totals.index))
            ],
            "trends": {
                "income": income_trend,
                "expenses": expense_trend,
                "savings_rate": (
                    round(avg_savings_rate * 100, 1) if avg_savings_rate else 0
                ),
            },
            "recurring_items": recurring,
            "insights": insights,
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import sys
from models.anomaly import AnomalyDetector
from services.anomaly_service import AnomalyService


def generate_anomaly_data(n_samples=1000, anomaly_ratio=0.05):
    """Generate synthetic transaction data with some anomalies"""

    # Define categories and their properties
    categories = {
        "groceries": {"mean": 100, "std": 20},
        "dining": {"mean": 50, "std": 15},
        "utilities": {"mean": 150, "std": 30},
        "rent": {"mean": 1200, "std": 100},
        "entertainment": {"mean": 80, "std": 25},
        "transportation": {"mean": 60, "std": 15},
    }

    # Generate transactions
    transactions = []
    today = datetime.now()

    # Generate normal transactions
    normal_count = int(n_samples * (1 - anomaly_ratio))
    for i in range(normal_count):
        # Random category
        category = random.choice(list(categories.keys()))
        category_props = categories[category]

        # Random amount based on category properties
        amount = -abs(
            random.normalvariate(category_props["mean"], category_props["std"])
        )

        # Random date within the last 180 days
        days_ago = random.randint(0, 180)
        date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        # Add transaction
        transactions.append(
            {
                "date": date,
                "amount": round(amount, 2),
                "category": category,
                "description": f"{category.title()} Payment",
            }
        )

    # Generate anomalous transactions
    anomaly_count = int(n_samples * anomaly_ratio)
    for i in range(anomaly_count):
        # Random category
        category = random.choice(list(categories.keys()))
        category_props = categories[category]

        # Create different types of anomalies
        anomaly_type = random.choice(["amount", "day", "combined"])

        if anomaly_type == "amount":
            # Large amount anomaly (3-10x normal)
            multiplier = random.uniform(3, 10)
            amount = -abs(
                random.normalvariate(category_props["mean"], category_props["std"])
                * multiplier
            )
            days_ago = random.randint(0, 180)
            date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        elif anomaly_type == "day":
            # Transaction on unusual day (weekend for rent/utilities)
            amount = -abs(
                random.normalvariate(category_props["mean"], category_props["std"])
            )

            if category in ["rent", "utilities"]:
                # Force a weekend date
                days_ago = random.randint(0, 180)
                base_date = today - timedelta(days=days_ago)
                # Adjust to nearest Saturday or Sunday
                weekday = base_date.weekday()
                if weekday < 5:  # Not yet weekend
                    days_to_weekend = 5 - weekday  # Days until Saturday
                    date = (base_date + timedelta(days=days_to_weekend)).strftime(
                        "%Y-%m-%d"
                    )
                else:
                    date = base_date.strftime("%Y-%m-%d")
            else:
                days_ago = random.randint(0, 180)
                date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        else:  # combined
            # Both unusual amount and unusual day
            multiplier = random.uniform(3, 10)
            amount = -abs(
                random.normalvariate(category_props["mean"], category_props["std"])
                * multiplier
            )

            if category in ["rent", "utilities"]:
                # Force a weekend date
                days_ago = random.randint(0, 180)
                base_date = today - timedelta(days=days_ago)
                # Adjust to nearest Saturday or Sunday
                weekday = base_date.weekday()
                if weekday < 5:  # Not yet weekend
                    days_to_weekend = 5 - weekday  # Days until Saturday
                    date = (base_date + timedelta(days=days_to_weekend)).strftime(
                        "%Y-%m-%d"
                    )
                else:
                    date = base_date.strftime("%Y-%m-%d")
            else:
                days_ago = random.randint(0, 180)
                date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        # Add anomalous transaction
        transactions.append(
            {
                "date": date,
                "amount": round(amount, 2),
                "category": category,
                "description": f"{category.title()} Payment (Unusual)",
            }
        )

    # Add some income transactions
    for i in range(int(n_samples * 0.1)):
        amount = random.uniform(1000, 5000)
        days_ago = random.randint(0, 180)
        date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        transactions.append(
            {
                "date": date,
                "amount": round(amount, 2),
                "category": "income",
                "description": "Salary Deposit",
            }
        )

    return pd.DataFrame(transactions)


def test_anomaly_detection():
    """Test the anomaly detection model"""
    print("\n===== Testing Anomaly Detection Model =====")

    # Generate test data
    print("Generating synthetic test data with anomalies...")
    df = generate_anomaly_data(n_samples=1000, anomaly_ratio=0.05)
    print(f"Generated {len(df)} test transactions")

    # Train the model
    print("\nTraining the anomaly detection model...")
    detector = AnomalyDetector()
    detector.fit(df)

    # Detect anomalies
    print("\nDetecting anomalies...")
    result_df = detector.detect_anomalies(df)

    # Print anomaly statistics
    anomalies = result_df[result_df["is_anomaly"]]
    print(
        f"\nDetected {len(anomalies)} anomalies ({len(anomalies) / len(df) * 100:.1f}% of transactions)"
    )

    # Print anomaly types breakdown
    anomaly_types = anomalies["anomaly_type"].value_counts()
    print("\nAnomaly Types:")
    for anomaly_type, count in anomaly_types.items():
        print(f"  {anomaly_type}: {count} ({count / len(anomalies) * 100:.1f}%)")

    # Print sample anomalies
    print("\nSample Anomalies:")
    for i, (_, row) in enumerate(anomalies.sample(min(5, len(anomalies))).iterrows()):
        print(f"\nAnomaly #{i + 1}:")
        print(f"  Category: {row.get('category', 'unknown')}")
        print(f"  Amount: ${abs(row['amount']):.2f}")
        print(
            f"  Date: {row['date'] if isinstance(row['date'], str) else row['date'].strftime('%Y-%m-%d')}"
        )
        print(f"  Type: {row['anomaly_type']}")
        print(f"  Confidence: {row['anomaly_confidence'] * 100:.1f}%")
        print(f"  Explanation: {row['anomaly_explanation']}")

    # Test the service layer
    print("\n===== Testing Anomaly Service =====")
    service = AnomalyService()

    # Train the service model
    print("Training service model...")
    result = service.train_anomaly_model(df.to_dict("records"))
    print(f"Training result: {result['success']}")
    if result["success"]:
        print(f"Model ID: {result['model_id']}")

    # Test service detection
    print("\nDetecting anomalies via service...")
    detection_result = service.detect_anomalies(
        user_id="test_user", transactions=df.to_dict("records")
    )

    # Print detection summary
    if "summary" in detection_result:
        summary = detection_result["summary"]
        print("\nAnomaly Detection Summary:")
        print(f"  Total Transactions: {summary['total_transactions']}")
        print(
            f"  Anomaly Count: {summary['anomaly_count']} ({summary['anomaly_percentage']}%)"
        )
        print(f"  Categories with Anomalies: {summary['categories_with_anomalies']}")

    # Print sample anomalies from service
    if "anomalies" in detection_result and detection_result["anomalies"]:
        print("\nSample Service-Detected Anomalies:")
        sample_anomalies = random.sample(
            detection_result["anomalies"], min(5, len(detection_result["anomalies"]))
        )

        for i, anomaly in enumerate(sample_anomalies):
            print(f"\nAnomaly #{i + 1}:")
            print(f"  Category: {anomaly['category']}")
            print(f"  Amount: ${abs(anomaly['amount']):.2f}")
            print(f"  Date: {anomaly['date']}")
            print(f"  Type: {anomaly['anomaly_type']}")
            print(f"  Confidence: {anomaly['anomaly_confidence'] * 100:.1f}%")
            print(f"  Explanation: {anomaly['explanation']}")

    print("\n===== Anomaly Detection Testing Complete =====")


if __name__ == "__main__":
    test_anomaly_detection()

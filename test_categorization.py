import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import sys
from models.categorization import TransactionCategorizer
from services.categorize_service import CategorizationService
from util.data_processing import extract_transaction_features


# Generate synthetic test data
def generate_test_data(n_samples=1000):
    """Generate synthetic transaction data for testing"""

    # Define categories and their common descriptions
    categories = {
        "groceries": [
            "Walmart",
            "Kroger",
            "Albertsons",
            "Safeway",
            "Trader Joe's",
            "Whole Foods",
            "Grocery Store",
        ],
        "dining": [
            "Restaurant",
            "McDonald's",
            "Starbucks",
            "Chipotle",
            "Subway",
            "Coffee Shop",
            "Pizza",
        ],
        "utilities": [
            "Electric Bill",
            "Water Bill",
            "Gas Bill",
            "Internet",
            "Phone Bill",
            "Cable",
            "Utilities",
        ],
        "transportation": [
            "Uber",
            "Lyft",
            "Gas Station",
            "Bus Fare",
            "Train Ticket",
            "Parking",
            "Car Rental",
        ],
        "shopping": [
            "Amazon",
            "Target",
            "Best Buy",
            "Macy's",
            "Home Depot",
            "Clothing Store",
            "Electronics",
        ],
        "entertainment": [
            "Movie Theater",
            "Netflix",
            "Spotify",
            "Concert Tickets",
            "Game Purchase",
            "Hulu",
            "Disney+",
        ],
        "housing": [
            "Rent Payment",
            "Mortgage",
            "Home Insurance",
            "HOA Fee",
            "Property Tax",
            "Furniture",
            "Home Repair",
        ],
        "healthcare": [
            "Doctor Visit",
            "Pharmacy",
            "Hospital",
            "Dental",
            "Vision",
            "Health Insurance",
            "Medical",
        ],
        "income": [
            "Salary Deposit",
            "Direct Deposit",
            "Paycheck",
            "Venmo Payment",
            "Bank Transfer",
            "Refund",
            "Interest",
        ],
    }

    # Generate random transactions
    transactions = []

    for _ in range(n_samples):
        # Random category
        category = random.choice(list(categories.keys()))

        # Random description based on category
        if category == "income":
            description = random.choice(categories[category])
            amount = round(random.uniform(500, 5000), 2)  # Positive amount for income
        else:
            description = random.choice(categories[category])
            amount = -round(random.uniform(1, 500), 2)  # Negative amount for expenses

        # Random date within the last 6 months
        days_ago = random.randint(0, 180)
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        # Add some noise to descriptions
        if random.random() < 0.3:
            description = f"{description} #{random.randint(1000, 9999)}"

        if random.random() < 0.2:
            description = f"POS {description}"

        transactions.append(
            {
                "description": description,
                "amount": amount,
                "date": date,
                "category": category,
            }
        )

    return pd.DataFrame(transactions)


def test_categorization_model():
    """Test the transaction categorization model"""
    print("\n===== Testing Transaction Categorization Model =====")

    # Generate test data
    print("Generating synthetic test data...")
    df = generate_test_data(n_samples=1000)
    print(f"Generated {len(df)} test transactions")

    # Split into training and testing sets
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    print(f"Training set: {len(train_df)} transactions")
    print(f"Testing set: {len(test_df)} transactions")

    # Extract features
    print("\nExtracting features...")
    train_df = extract_transaction_features(train_df)
    test_df = extract_transaction_features(test_df)

    # Train the model
    print("\nTraining the categorization model...")
    categorizer = TransactionCategorizer()
    categorizer.fit(train_df)

    # Test on individual transactions
    print("\nTesting individual transaction predictions:")
    test_examples = [
        {"description": "Walmart Groceries", "amount": -78.45, "date": "2025-03-01"},
        {"description": "Starbucks Coffee", "amount": -5.67, "date": "2025-03-02"},
        {"description": "Netflix Subscription", "amount": -14.99, "date": "2025-03-03"},
        {"description": "Salary Deposit", "amount": 3500.00, "date": "2025-03-04"},
        {"description": "Unknown Transaction", "amount": -25.00, "date": "2025-03-05"},
    ]

    for example in test_examples:
        example_df = pd.DataFrame([example])
        example_df = extract_transaction_features(example_df)

        result = categorizer.predict(example_df)[0]
        print(f"\nTransaction: {example['description']}, ${example['amount']}")
        print(
            f"Predicted category: {result['category']} (confidence: {result['confidence']:.2f})"
        )
        print("Alternative categories:")
        for alt in result["alternative_categories"]:
            print(f"  - {alt['category']} (confidence: {alt['confidence']:.2f})")

    # Test the service layer
    print("\n===== Testing Categorization Service =====")
    service = CategorizationService()

    # Train the service model
    print("Training service model...")
    result = service.train_model(train_df.to_dict("records"))
    print(f"Training result: {result['success']}")
    print(f"Model ID: {result['model_id']}")

    # Test service predictions
    print("\nTesting service predictions:")
    for example in test_examples:
        result = service.categorize_transaction(
            description=example["description"],
            amount=example["amount"],
            date=example["date"],
        )
        print(f"\nTransaction: {example['description']}, ${example['amount']}")
        print(
            f"Predicted category: {result['category']} (confidence: {result['confidence']:.2f})"
        )
        print("Alternative categories:")
        for alt in result["alternative_categories"]:
            print(f"  - {alt['category']} (confidence: {alt['confidence']:.2f})")

    print("\n===== Categorization Testing Complete =====")


if __name__ == "__main__":
    test_categorization_model()

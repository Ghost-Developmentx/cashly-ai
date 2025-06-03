"""
Base class for synthetic data generation.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List
import random


class BaseSyntheticGenerator:
    """Base class for generating synthetic financial data."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Common categories
        self.expense_categories = {
            'Food & Dining': (10, 150, 0.25),  # (min, max, frequency)
            'Transportation': (5, 100, 0.15),
            'Shopping': (20, 300, 0.20),
            'Entertainment': (10, 200, 0.10),
            'Bills & Utilities': (50, 300, 0.10),
            'Healthcare': (20, 500, 0.05),
            'Travel': (100, 2000, 0.05),
            'Groceries': (50, 200, 0.10)
        }

        self.income_categories = {
            'Salary': (3000, 5000, 0.70),
            'Freelance': (500, 2000, 0.20),
            'Investment': (100, 1000, 0.10)
        }

        # Merchant templates
        self.merchants = {
            'Food & Dining': ['Starbucks', 'McDonalds', 'Chipotle', 'Pizza Hut', 'Subway'],
            'Transportation': ['Uber', 'Shell Gas', 'Chevron', 'Metro Transit', 'Lyft'],
            'Shopping': ['Amazon', 'Walmart', 'Target', 'Best Buy', 'Home Depot'],
            'Entertainment': ['Netflix', 'AMC Theater', 'Spotify', 'Steam', 'PlayStation'],
            'Groceries': ['Whole Foods', 'Kroger', 'Safeway', 'Trader Joes', 'Costco']
        }

    @staticmethod
    def generate_date_range(
            start_date: datetime,
            end_date: datetime
    ) -> List[datetime]:
        """Generate list of dates in range."""
        days = (end_date - start_date).days
        return [start_date + timedelta(days=i) for i in range(days + 1)]

    @staticmethod
    def generate_transaction_description(
            category: str,
            merchant: str
    ) -> str:
        """Generate realistic transaction description."""
        templates = [
            f"{merchant} PURCHASE",
            f"{merchant} - ONLINE",
            f"POS TRANSACTION - {merchant}",
            f"{merchant} #{random.randint(1000, 9999)}",
            f"{merchant} STORE #{random.randint(1, 999)}"
        ]
        return random.choice(templates)

    @staticmethod
    def add_noise(value: float, noise_level: float = 0.1) -> float:
        """Add random noise to a value."""
        noise = np.random.normal(0, value * noise_level)
        return max(0, value + noise)

    @staticmethod
    def add_seasonality(
            date: datetime,
            base_amount: float,
            category: str
    ) -> float:
        """Add seasonal variations to amounts."""
        month = date.month

        # Holiday season boost for shopping
        if category == 'Shopping' and month in [11, 12]:
            return base_amount * 1.5

        # Summer travel boost
        if category == 'Travel' and month in [6, 7, 8]:
            return base_amount * 1.3

        # Winter utility bills
        if category == 'Bills & Utilities' and month in [12, 1, 2]:
            return base_amount * 1.2

        return base_amount
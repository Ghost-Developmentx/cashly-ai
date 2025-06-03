"""
Generate synthetic transaction data for model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import random

from .base_generator import BaseSyntheticGenerator


class TransactionGenerator(BaseSyntheticGenerator):
    """Generate synthetic transaction data."""

    def generate_transactions(
            self,
            num_days: int = 365,
            transactions_per_day: Tuple[int, int] = (1, 5),
            anomaly_rate: float = 0.02
    ) -> pd.DataFrame:
        """Generate synthetic transactions with realistic patterns."""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)

        transactions = []

        # Generate daily transactions
        for date in self.generate_date_range(start_date, end_date):
            # Determine the number of transactions for this day
            num_trans = random.randint(*transactions_per_day)

            # Weekend adjustment
            if date.weekday() >= 5:  # Weekend
                num_trans = int(num_trans * 1.3)

            for _ in range(num_trans):
                # Decide if income or expense
                is_income = random.random() < 0.1  # 10% income

                if is_income:
                    transaction = self._generate_income_transaction(date)
                    # Skip if transaction is None (e.g., salary on the wrong day)
                    if transaction is None:
                        continue
                else:
                    transaction = self._generate_expense_transaction(date)

                # Add anomaly
                if random.random() < anomaly_rate:
                    transaction = self._add_anomaly(transaction)

                transactions.append(transaction)

        # Add recurring transactions
        transactions.extend(self._generate_recurring_transactions(start_date, end_date))

        df = pd.DataFrame(transactions)
        return df.sort_values('date').reset_index(drop=True)

    def _generate_expense_transaction(self, date: datetime) -> Dict[str, Any]:
        """Generate a single expense transaction."""
        # Choose category based on weights
        categories = list(self.expense_categories.keys())
        weights = [self.expense_categories[c][2] for c in categories]
        category = np.random.choice(categories, p=weights)

        # Get amount range
        min_amt, max_amt, _ = self.expense_categories[category]
        base_amount = random.uniform(min_amt, max_amt)

        # Apply seasonality
        amount = self.add_seasonality(date, base_amount, category)

        # Add noise
        amount = self.add_noise(amount, 0.15)

        # Choose merchant
        merchant = random.choice(self.merchants.get(category, ['Generic Store']))

        return {
            'date': date,
            'amount': -round(amount, 2),  # Negative for expense
            'category': category,
            'description': self.generate_transaction_description(category, merchant),
            'merchant': merchant,
            'type': 'expense'
        }

    def _generate_income_transaction(self, date: datetime) -> Dict[str, Any]:
        """Generate a single income transaction."""
        categories = list(self.income_categories.keys())
        weights = [self.income_categories[c][2] for c in categories]
        category = np.random.choice(categories, p=weights)

        min_amt, max_amt, _ = self.income_categories[category]
        amount = random.uniform(min_amt, max_amt)

        # Salary typically on specific days
        if category == 'Salary':
            # Biweekly on 15th and last day
            if date.day not in [15, 30, 31]:
                return None

        return {
            'date': date,
            'amount': round(amount, 2),
            'category': category,
            'description': f"{category} Deposit",
            'merchant': category,
            'type': 'income'
        }

    @staticmethod
    def _generate_recurring_transactions(
            start_date: datetime,
            end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Generate recurring subscriptions and bills."""
        recurring = []

        # Define recurring transactions
        recurring_items = [
            ('Netflix Subscription', 15.99, 'Entertainment', 1),
            ('Spotify Premium', 9.99, 'Entertainment', 1),
            ('Rent Payment', 1500, 'Bills & Utilities', 1),
            ('Internet Bill', 79.99, 'Bills & Utilities', 15),
            ('Gym Membership', 49.99, 'Healthcare', 5),
        ]

        for desc, amount, category, day_of_month in recurring_items:
            current = start_date.replace(day=min(day_of_month, 28))

            while current <= end_date:
                recurring.append({
                    'date': current,
                    'amount': -amount,
                    'category': category,
                    'description': desc,
                    'merchant': desc.split()[0],
                    'type': 'expense'
                })

                # Next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)

        return recurring

    @staticmethod
    def _add_anomaly(transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Add anomalous characteristics to a transaction."""
        anomaly_type = random.choice(['amount', 'time', 'category'])

        if anomaly_type == 'amount':
            # Make amount 5-10x larger
            transaction['amount'] = transaction['amount'] * random.uniform(5, 10)
        elif anomaly_type == 'time':
            # Change to unusual hour
            transaction['date'] = transaction['date'].replace(
                hour=random.choice([3, 4, 5, 23])
            )

        return transaction
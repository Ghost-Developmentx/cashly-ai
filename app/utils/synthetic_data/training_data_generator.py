"""
Generate training data for all ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

from .transaction_generator import TransactionGenerator

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generate training data for all model types."""

    def __init__(self, seed: int = 42):
        self.transaction_gen = TransactionGenerator(seed)

    def generate_all_training_data(
            self,
            num_days: int = 365,
            num_users: int = 1
    ) -> Dict[str, pd.DataFrame]:
        """Generate training data for all models."""

        logger.info(f"Generating {num_days} days of data for {num_users} users")

        all_data = {}

        for user_id in range(num_users):
            # Generate base transaction data
            transactions = self.transaction_gen.generate_transactions(
                num_days=num_days,
                transactions_per_day=(2, 8),
                anomaly_rate=0.03
            )

            # Add user_id
            transactions['user_id'] = f'user_{user_id}'

            if user_id == 0:
                all_data['transactions'] = transactions
            else:
                all_data['transactions'] = pd.concat(
                    [all_data['transactions'], transactions],
                    ignore_index=True
                )

        # Generate specific datasets for each model
        all_data['categorization'] = self._prepare_categorization_data(
            all_data['transactions']
        )
        all_data['forecasting'] = self._prepare_forecasting_data(
            all_data['transactions']
        )
        all_data['anomaly'] = self._prepare_anomaly_data(
            all_data['transactions']
        )
        all_data['budgeting'] = self._prepare_budgeting_data(
            all_data['transactions']
        )
        all_data['trend'] = self._prepare_trend_data(
            all_data['transactions']
        )

        logger.info(f"Generated data summary:")
        for key, df in all_data.items():
            logger.info(f"  {key}: {len(df)} rows")

        return all_data

    @staticmethod
    def _prepare_categorization_data(
            transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare data for categorization model."""
        df = transactions.copy()

        # Ensure we have good category distribution
        category_counts = df['category'].value_counts()
        logger.info(f"Category distribution:\n{category_counts}")

        # Add some mislabeled data for robustness
        mislabel_idx = df.sample(frac=0.05).index
        categories = df['category'].unique()

        for idx in mislabel_idx:
            # Swap to random different category
            current = df.loc[idx, 'category']
            new_cat = np.random.choice(
                [c for c in categories if c != current]
            )
            df.loc[idx, 'category'] = new_cat

        return df

    @staticmethod
    def _prepare_forecasting_data(
            transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare data for forecasting model."""
        # Aggregate by date for time series
        daily = transactions.groupby('date').agg({
            'amount': 'sum',
            'category': 'count'
        }).rename(columns={'category': 'transaction_count'})

        # Add missing dates
        date_range = pd.date_range(
            start=transactions['date'].min(),
            end=transactions['date'].max(),
            freq='D'
        )

        daily = daily.reindex(date_range, fill_value=0)
        daily.index.name = 'date'

        return daily.reset_index()

    @staticmethod
    def _prepare_anomaly_data(
            transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare data for anomaly detection."""
        df = transactions.copy()

        # Label anomalies (for evaluation)
        df['is_anomaly'] = 0

        # Mark transactions with extreme amounts
        amount_mean = df['amount'].abs().mean()
        amount_std = df['amount'].abs().std()
        threshold = amount_mean + 3 * amount_std

        df.loc[df['amount'].abs() > threshold, 'is_anomaly'] = 1

        # Mark late night transactions
        df['hour'] = pd.to_datetime(df['date']).dt.hour
        df.loc[df['hour'].isin([0, 1, 2, 3, 4, 5]), 'is_anomaly'] = 1

        logger.info(f"Anomaly rate: {df['is_anomaly'].mean():.2%}")

        return df

    @staticmethod
    def _prepare_budgeting_data(
            transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare data for budget recommendation."""
        # Include all transactions as-is
        # The model will separate income/expenses internally
        return transactions.copy()

    @staticmethod
    def _prepare_trend_data(
            transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare data for trend analysis."""
        # Similar to forecasting but keep more detail
        return transactions.copy()
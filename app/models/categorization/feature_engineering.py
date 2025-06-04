"""
Feature engineering for transaction categorization.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models.base import BaseTransformer

class TextFeatureExtractor(BaseTransformer):
    """Extracts text features from transaction descriptions."""

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        self.feature_names = []

    def fit(self, X: pd.DataFrame) -> "TextFeatureExtractor":
        """Fit the text vectorizer."""
        if 'description' in X.columns:
            cleaned_text = X['description'].apply(self._clean_text)

            # Check if all text is empty after cleaning
            non_empty_text = cleaned_text[cleaned_text.str.len() > 0]
            if len(non_empty_text) == 0:
                raise ValueError("No features extracted")

            try:
                self.vectorizer.fit(cleaned_text)
                self.feature_names = self.vectorizer.get_feature_names_out().tolist()
            except ValueError as e:
                if "empty vocabulary" in str(e):
                    raise ValueError("No features extracted")
                raise
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform descriptions to TF-IDF features."""
        if 'description' not in X.columns:
            return X

        # Clean and vectorize text
        cleaned_text = X['description'].apply(self._clean_text)
        tfidf_matrix = self.vectorizer.transform(cleaned_text)

        # Convert to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{name}' for name in self.feature_names],
            index=X.index
        )

        # Concatenate with original data
        result = pd.concat([X, tfidf_df], axis=1)

        return result

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean transaction description."""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove common transaction prefixes/suffixes
        patterns_to_remove = [
            r'\b\d{4,}\b',  # Remove long numbers
            r'\bpos\b',
            r'\bpurchase\b',
            r'\btransaction\b',
            r'\bcard\b',
            r'\bending\b',
            r'\b\d{2}/\d{2}\b',  # Dates
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text)

        return text.strip()

class MerchantFeatureExtractor(BaseTransformer):
    """Extracts merchant-based features."""

    def __init__(self):
        self.merchant_patterns = {
            'restaurant': ['restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'sushi', 'grill'],
            'grocery': ['grocery', 'supermarket', 'market', 'whole foods', 'trader joe'],
            'gas': ['shell', 'exxon', 'chevron', 'gas station', 'fuel'],
            'online': ['amazon', 'ebay', 'etsy', 'online', 'digital'],
            'subscription': ['netflix', 'spotify', 'hulu', 'subscription', 'monthly'],
            'travel': ['airline', 'hotel', 'airbnb', 'uber', 'lyft', 'taxi'],
            'retail': ['walmart', 'target', 'costco', 'store', 'mall'],
            'utility': ['electric', 'gas', 'water', 'internet', 'phone']
        }

    def fit(self, X: pd.DataFrame) -> "MerchantFeatureExtractor":
        """No fitting needed for rule-based extraction."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract merchant features."""
        df = X.copy()

        # Extract merchant type features
        for merchant_type, patterns in self.merchant_patterns.items():
            df[f'merchant_{merchant_type}'] = df['description'].apply(
                lambda x: self._contains_pattern(x, patterns)
            ).astype(int)

        # Extract merchant name (first significant word)
        df['merchant_name'] = df['description'].apply(self._extract_merchant_name)

        # Merchant frequency encoding
        merchant_counts = df['merchant_name'].value_counts().to_dict()
        df['merchant_frequency'] = df['merchant_name'].map(merchant_counts)

        return df

    @staticmethod
    def _contains_pattern(text: str, patterns: List[str]) -> bool:
        """Check if text contains any of the patterns."""
        if pd.isna(text):
            return False

        text_lower = str(text).lower()
        return any(pattern in text_lower for pattern in patterns)

    @staticmethod
    def _extract_merchant_name(text: str) -> str:
        """Extract likely merchant name from description."""
        if pd.isna(text):
            return "unknown"

        # Clean and split text
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        words = text.split()

        # Skip common transaction words
        skip_words = {'pos', 'purchase', 'transaction', 'card', 'debit', 'credit'}

        # Find the first significant word
        for word in words:
            if len(word) > 3 and word not in skip_words:
                return word

        return "unknown"

class AmountFeatureExtractor(BaseTransformer):
    """Extracts amount-based features for categorization."""

    def __init__(self):
        self.amount_bins = [0, 10, 25, 50, 100, 250, 500, 1000, float('inf')]
        self.amount_labels = ['tiny', 'small', 'medium', 'large', 'xlarge', 'huge', 'massive', 'extreme']

    def fit(self, X: pd.DataFrame) -> "AmountFeatureExtractor":
        """No fitting needed."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract amount features."""
        df = X.copy()

        if 'amount' not in df.columns:
            return df

        # Absolute amount
        df['amount_abs'] = df['amount'].abs()

        # Amount categories
        df['amount_category'] = pd.cut(
            df['amount_abs'],
            bins=self.amount_bins,
            labels=self.amount_labels
        )

        # One-hot encode amount categories
        amount_dummies = pd.get_dummies(df['amount_category'], prefix='amount_cat')
        df = pd.concat([df, amount_dummies], axis=1)

        # Log amount (for scaling)
        df['amount_log'] = np.log1p(df['amount_abs'])

        # Is expense or income
        df['is_expense'] = (df['amount'] < 0).astype(int)
        df['is_income'] = (df['amount'] > 0).astype(int)

        # Round amount indicators
        df['is_round_amount'] = (df['amount_abs'] % 10 == 0).astype(int)
        df['is_round_hundred'] = (df['amount_abs'] % 100 == 0).astype(int)

        return df
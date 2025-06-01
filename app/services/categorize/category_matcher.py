"""
Pattern matching for transaction categorization.
"""

import logging
import re
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class CategoryMatcher:
    """
    A utility class for categorizing transaction descriptions into predefined categories.

    This class is designed to process transaction descriptions and categorize them into
    defined categories like "Food & Dining", "Transportation", and more. It uses regular
    expressions to match patterns in descriptions. Users can also add custom patterns to
    extend categorization rules.

    Attributes
    ----------
    patterns : dict
        A dictionary where keys are category names (str) and values are lists of regex
        patterns (str) associated with those categories.
    compiled_patterns : dict
        A dictionary where keys are category names (str) and values are lists of compiled
        regex patterns (regex objects) corresponding to the categories.
    """

    def __init__(self):
        # Define category patterns
        self.patterns = {
            "Food & Dining": [
                r"restaurant",
                r"cafe",
                r"coffee",
                r"pizza",
                r"burger",
                r"grocery",
                r"market",
                r"food",
                r"dining",
                r"eat",
                r"uber.*eats",
                r"doordash",
                r"grubhub",
                r"seamless",
            ],
            "Transportation": [
                r"uber(?!.*eats)",
                r"lyft",
                r"taxi",
                r"gas",
                r"fuel",
                r"parking",
                r"toll",
                r"transit",
                r"metro",
                r"train",
            ],
            "Shopping": [
                r"amazon",
                r"walmart",
                r"target",
                r"store",
                r"shop",
                r"mall",
                r"retail",
                r"clothing",
                r"shoes",
            ],
            "Entertainment": [
                r"movie",
                r"theater",
                r"cinema",
                r"concert",
                r"ticket",
                r"game",
                r"sport",
                r"gym",
                r"fitness",
            ],
            "Bills & Utilities": [
                r"electric",
                r"gas",
                r"water",
                r"internet",
                r"phone",
                r"utilities",
                r"cable",
                r"insurance",
            ],
            "Healthcare": [
                r"doctor",
                r"hospital",
                r"clinic",
                r"pharmacy",
                r"medical",
                r"dental",
                r"health",
                r"prescription",
            ],
            "Travel": [
                r"airline",
                r"hotel",
                r"airbnb",
                r"flight",
                r"travel",
                r"vacation",
                r"booking",
            ],
            "Subscriptions": [
                r"netflix",
                r"spotify",
                r"hulu",
                r"disney",
                r"subscription",
                r"monthly",
                r"membership",
            ],
        }

        # Compile patterns
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.patterns.items()
        }

    async def find_best_match(self, description: str) -> Optional[str]:
        """
        Find the best category match for a description.

        Args:
            description: Transaction description

        Returns:
            Matched category or None
        """
        if not description:
            return None

        description_lower = description.lower()
        scores = {}

        # Score each category
        for category, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(description_lower):
                    score += 1

            if score > 0:
                scores[category] = score

        # Return category with highest score
        if scores:
            best_category = max(scores, key=scores.get)
            return best_category

        return None

    async def add_custom_pattern(self, category: str, pattern: str) -> bool:
        """Add a custom pattern for a category."""
        try:
            if category not in self.patterns:
                self.patterns[category] = []
                self.compiled_patterns[category] = []

            self.patterns[category].append(pattern)
            self.compiled_patterns[category].append(re.compile(pattern, re.IGNORECASE))

            logger.info(f"Added pattern '{pattern}' to category '{category}'")
            return True

        except Exception as e:
            logger.error(f"Failed to add pattern: {e}")
            return False

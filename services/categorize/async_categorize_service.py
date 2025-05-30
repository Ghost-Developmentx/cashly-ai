"""
Async transaction categorization service.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio

from .category_matcher import CategoryMatcher
from .ml_categorizer import MLCategorizer
from .category_rules import CategoryRules

logger = logging.getLogger(__name__)


class AsyncCategorizationService:
    """
    Asynchronous service for categorizing transactions.

    This service provides functionalities for categorizing single or multiple
    financial transactions based on various methods, including rule-based
    matching, machine learning, and pattern matching. It also supports updating
    its knowledge base through user feedback and generating statistics about
    categorizations.

    Attributes
    ----------
    matcher : CategoryMatcher
        Instance of `CategoryMatcher` used for pattern matching.
    ml_categorizer : MLCategorizer
        Instance of `MLCategorizer` used for machine learning-based
        transaction categorization.
    rules : CategoryRules
        Instance of `CategoryRules` used for rule-based categorization.
    batch_size : int
        Number of transactions to process in a single batch when
        categorizing in bulk.
    """

    def __init__(self):
        self.matcher = CategoryMatcher()
        self.ml_categorizer = MLCategorizer()
        self.rules = CategoryRules()
        self.batch_size = 100

    async def categorize_transaction(
        self, description: str, amount: float, merchant: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Categorize a single transaction.

        Args:
            description: Transaction description
            amount: Transaction amount
            merchant: Optional merchant name

        Returns:
            Category and confidence
        """
        try:
            # Try rule-based matching first
            rule_category = await self.rules.match_category(description, merchant)

            if rule_category:
                return {
                    "category": rule_category,
                    "confidence": 0.95,
                    "method": "rules",
                }

            # Try ML categorization
            ml_result = await self.ml_categorizer.categorize(description, amount)

            if ml_result["confidence"] > 0.7:
                return ml_result

            # Fallback to pattern matching
            pattern_category = await self.matcher.find_best_match(description)

            return {
                "category": pattern_category or "Other",
                "confidence": 0.5 if pattern_category else 0.3,
                "method": "pattern",
            }

        except Exception as e:
            logger.error(f"Categorization failed: {e}")
            return {"category": "Uncategorized", "confidence": 0.0, "error": str(e)}

    async def categorize_batch(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Categorize multiple transactions.

        Args:
            transactions: List of transactions

        Returns:
            Categorization results
        """
        results = []

        # Process in batches for efficiency
        for i in range(0, len(transactions), self.batch_size):
            batch = transactions[i : i + self.batch_size]

            # Process batch concurrently
            tasks = [
                self.categorize_transaction(
                    txn.get("description", ""),
                    float(txn.get("amount", 0)),
                    txn.get("merchant"),
                )
                for txn in batch
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    async def learn_from_feedback(
        self, description: str, amount: float, correct_category: str
    ) -> bool:
        """
        Learn from user feedback.

        Args:
            description: Transaction description
            amount: Transaction amount
            correct_category: User-provided category

        Returns:
            Success status
        """
        try:
            # Update rules
            await self.rules.add_user_rule(description, correct_category)

            # Update ML model (if applicable)
            await self.ml_categorizer.update_training(
                description, amount, correct_category
            )

            logger.info(
                f"Learned new categorization: {description} -> {correct_category}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to learn from feedback: {e}")
            return False

    @staticmethod
    async def get_category_statistics(
        transactions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get categorization statistics.

        Args:
            transactions: List of transactions

        Returns:
            Category statistics
        """
        categorized = 0
        uncategorized = 0
        category_counts = {}

        for txn in transactions:
            category = txn.get("category", "Uncategorized")

            if category and category != "Uncategorized":
                categorized += 1
                category_counts[category] = category_counts.get(category, 0) + 1
            else:
                uncategorized += 1

        total = len(transactions)

        return {
            "total_transactions": total,
            "categorized": categorized,
            "uncategorized": uncategorized,
            "categorization_rate": ((categorized / total * 100) if total > 0 else 0),
            "category_distribution": category_counts,
            "top_categories": sorted(
                category_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

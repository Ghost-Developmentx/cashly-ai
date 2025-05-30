"""
Rule-based categorization engine.
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class CategoryRules:
    """
    Encapsulates rules for transaction categorization.

    This class is used to determine the category of a financial transaction based on
    either predefined merchant-specific rules or user-defined rules. It provides methods
    for matching transactions, adding rules, removing rules, and retrieving all existing
    rules. This functionality is helpful in financial applications for tracking and
    categorizing expenses.

    Attributes
    ----------
    merchant_rules : dict
        Predefined rules mapping merchant names to categories.
    user_rules : dict
        User-defined rules mapping custom patterns to categories.
    """

    def __init__(self):
        # Merchant-specific rules
        self.merchant_rules = {
            "walmart": "Shopping",
            "whole foods": "Food & Dining",
            "shell": "Transportation",
            "netflix": "Subscriptions",
            "uber": "Transportation",
            "uber eats": "Food & Dining",
            "amazon": "Shopping",
            "starbucks": "Food & Dining",
        }

        # User-defined rules
        self.user_rules = {}

    async def match_category(
        self, description: str, merchant: Optional[str] = None
    ) -> Optional[str]:
        """
        Match category using rules.

        Args:
            description: Transaction description
            merchant: Merchant name if available

        Returns:
            Matched category or None
        """
        description_lower = description.lower() if description else ""
        merchant_lower = merchant.lower() if merchant else ""

        # Check user rules first
        for pattern, category in self.user_rules.items():
            if pattern in description_lower or pattern in merchant_lower:
                return category

        # Check merchant rules
        for merchant_pattern, category in self.merchant_rules.items():
            if (
                merchant_pattern in merchant_lower
                or merchant_pattern in description_lower
            ):
                return category

        return None

    async def add_user_rule(self, pattern: str, category: str) -> bool:
        """Add a user-defined rule."""
        try:
            self.user_rules[pattern.lower()] = category
            logger.info(f"Added user rule: '{pattern}' -> '{category}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add user rule: {e}")
            return False

    async def remove_user_rule(self, pattern: str) -> bool:
        """Remove a user-defined rule."""
        pattern_lower = pattern.lower()
        if pattern_lower in self.user_rules:
            del self.user_rules[pattern_lower]
            logger.info(f"Removed user rule: '{pattern}'")
            return True
        return False

    def get_all_rules(self) -> Dict[str, Dict[str, str]]:
        """Get all categorization rules."""
        return {"merchant_rules": self.merchant_rules, "user_rules": self.user_rules}

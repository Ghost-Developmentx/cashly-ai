import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

async def filter_by_date_range(
    transactions: List[Dict[str, Any]],
    start_date: datetime.date,
    end_date: datetime.date,
) -> List[Dict[str, Any]]:
    """Filter transactions by date range."""
    filtered = []
    for txn in transactions:
        try:
            txn_date = datetime.strptime(txn["date"], "%Y-%m-%d").date()
            if start_date <= txn_date <= end_date:
                filtered.append(txn)
        except ValueError:
            logger.warning(f"Invalid date format: {txn.get('date')}")
    return filtered


async def filter_by_categories(
    transactions: List[Dict[str, Any]], categories: List[str]
) -> List[Dict[str, Any]]:
    """Filter transactions by categories."""
    categories_lower = [c.lower() for c in categories]
    return [
        txn
        for txn in transactions
        if any(
            cat in str(txn.get("category", "")).lower() for cat in categories_lower
        )
    ]


async def calculate_category_totals(
    transactions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Calculate spending totals by category."""
    totals = {}
    for txn in transactions:
        if float(txn["amount"]) < 0:  # Only expenses
            category = txn.get("category", "Uncategorized")
            totals[category] = totals.get(category, 0) + abs(float(txn["amount"]))
    return totals
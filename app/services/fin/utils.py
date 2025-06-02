import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_date(date_val: Any) -> str:
    """
    Convert a date value to a string in YYYY-MM-DD format.

    Args:
        date_val: A string or datetime-like object.

    Returns:
        A formatted date string.

    Raises:
        ValueError if the input is not a valid date format.
    """
    if isinstance(date_val, str):
        return date_val[:10]
    if hasattr(date_val, "strftime"):
        return date_val.strftime("%Y-%m-%d")
    raise ValueError(f"Unrecognized date value: {date_val!r}")


def normalize_transaction_dates(transactions: Optional[List[dict]]) -> List[dict]:
    if not transactions:
        return []

    for txn in transactions:
        if "date" in txn and isinstance(txn["date"], str):
            try:
                txn["date"] = datetime.fromisoformat(txn["date"])
            except Exception:
                pass  # silently fail if invalid
    return transactions

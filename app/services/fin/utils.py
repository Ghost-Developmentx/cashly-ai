import logging
from typing import Any, Dict, List

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


def normalize_transaction_dates(
    transactions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ensure all transactions have a normalized 'date' field (YYYY-MM-DD).

    Args:
        transactions: List of transaction dictionaries.

    Returns:
        A new list of transactions with normalized 'date' fields.
    """
    normalized = []
    for txn in transactions:
        if "date" not in txn:
            continue
        try:
            txn_copy = dict(txn)
            txn_copy["date"] = parse_date(txn_copy["date"])
            normalized.append(txn_copy)
        except Exception as e:
            logger.warning(f"Skipping transaction with invalid date: {e}")
    return normalized

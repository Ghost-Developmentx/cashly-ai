from datetime import datetime, timedelta
from typing import Optional, Any


def get_due_date(due_date_str: Optional[str]) -> str:
    """Get or calculate the due date (default: 30 days from now)."""
    if due_date_str:
        try:
            # Validate date format
            parsed = datetime.strptime(due_date_str, "%Y-%m-%d")
            # Don't allow past due dates
            if parsed.date() < datetime.now().date():
                return (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            return due_date_str
        except ValueError:
            pass

    # Default: 30 days from now
    return (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")


def parse_date(date_str: Any) -> datetime.date:
    """Parse date string to a date object."""
    if not date_str:
        return datetime.max.date()

    try:
        if isinstance(date_str, str):
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        return datetime.max.date()
    except ValueError:
        return datetime.max.date()
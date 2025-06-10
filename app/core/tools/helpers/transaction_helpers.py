from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


async def filter_transactions(
        transactions: List[Dict[str, Any]],
        user_context: Dict[str, Any],
        days: int = 30,
        category: Optional[str] = None,
        account_id: Optional[str] = None,
        account_name: Optional[str] = None,
        transaction_type: str = "all"
) -> List[Dict[str, Any]]:
    """Apply filters to transactions."""
    filtered = transactions.copy()

    # Date filter
    if days and days > 0:
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered = [
            t for t in filtered
            if _parse_date(t.get("date")) >= cutoff_date.date()
        ]

    # Category filter
    if category:
        category_lower = category.lower()
        filtered = [
            t for t in filtered
            if t.get("category", "").lower() == category_lower
        ]

    # Account filter
    if account_id or account_name:
        account = await _resolve_account(user_context, account_id, account_name)
        if account:
            filtered = [
                t for t in filtered
                if str(t.get("account_id")) == str(account["id"])
            ]

    # Type filter
    if transaction_type != "all":
        if transaction_type == "income":
            filtered = [t for t in filtered if float(t.get("amount", 0)) > 0]
        elif transaction_type == "expense":
            filtered = [t for t in filtered if float(t.get("amount", 0)) < 0]

    # Sort by date (newest first)
    filtered.sort(
        key=lambda t: _parse_date(t.get("date", "")),
        reverse=True
    )

    return filtered

async def resolve_account(
        user_context: Dict[str, Any],
        account_id: Optional[str] = None,
        account_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Resolve account by ID or name."""
    accounts = user_context.get("accounts", [])

    if account_id:
        return next(
            (acc for acc in accounts if str(acc.get("id")) == str(account_id)),
            None
        )

    if account_name:
        account_name_clean = account_name.strip().lower()
        # Try exact match first
        account = next(
            (acc for acc in accounts
             if acc.get("name", "").lower() == account_name_clean),
            None
        )
        # Try partial match
        if not account:
            account = next(
                (acc for acc in accounts
                 if account_name_clean in acc.get("name", "").lower()),
                None
            )
        return account

    return None


async def prepare_updates(tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and prepare transaction updates."""
    updates = {}

    if "amount" in tool_args:
        try:
            amount = float(tool_args["amount"])
            if amount == 0:
                return {"error": "Transaction amount cannot be zero"}
            updates["amount"] = amount
        except (ValueError, TypeError):
            return {"error": "Invalid amount value"}

    if "description" in tool_args:
        desc = tool_args["description"].strip()
        if not desc:
            return {"error": "Description cannot be empty"}
        updates["description"] = desc

    if "category" in tool_args:
        updates["category"] = tool_args["category"].strip()

    if "date" in tool_args:
        date = _get_transaction_date(tool_args["date"])
        if not date:
            return {"error": "Invalid date format. Please use YYYY-MM-DD"}
        updates["date"] = date

    if "recurring" in tool_args:
        updates["recurring"] = bool(tool_args["recurring"])

    return updates


def calculate_summary(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate transaction summary statistics."""
    total_income = sum(
        float(t.get("amount", 0)) for t in transactions
        if float(t.get("amount", 0)) > 0
    )
    total_expenses = sum(
        abs(float(t.get("amount", 0))) for t in transactions
        if float(t.get("amount", 0)) < 0
    )

    return {
        "count": len(transactions),
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "net_change": round(total_income - total_expenses, 2)
    }


def get_category_breakdown(transactions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Get spending breakdown by category."""
    breakdown = {}

    for transaction in transactions:
        category = transaction.get("category", "Uncategorized")
        amount = abs(float(transaction.get("amount", 0)))

        if category in breakdown:
            breakdown[category] += amount
        else:
            breakdown[category] = amount

    # Round all values and sort by amount
    breakdown = {k: round(v, 2) for k, v in breakdown.items()}
    return dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))


def get_transaction_date(date_str: Optional[str]) -> Optional[str]:
    """Parse and validate transaction date."""
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")

    try:
        # Try to parse the date
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        # Try other common formats
        for fmt in ["%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y"]:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return None


def parse_date(date_str: Any) -> datetime.date:
    """Parse date string to date object."""
    if not date_str:
        return datetime.min.date()

    try:
        if isinstance(date_str, str):
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        return datetime.min.date()
    except ValueError:
        return datetime.min.date()
"""
Async handlers for transaction-related tools.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AsyncTransactionHandlers:
    """
    Handles asynchronous operations for managing user transactions.

    The `AsyncTransactionHandlers` class provides methods for retrieving, creating,
    updating, deleting, and categorizing user transactions in an asynchronous manner.
    It also supports filtering transactions based on various criteria, preparing updates
    for transactions, and performing operations such as calculating transaction summaries
    or categorizing uncategorized transactions.

    Attributes
    ----------
    No public attributes.
    """

    def __init__(self):
        pass  # No Rails client needed for these operations

    async def get_transactions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve and filter user transactions."""
        tool_args = context["tool_args"]
        user_context = context["user_context"]
        transactions = context["transactions"]

        try:
            # Apply filters
            filtered_transactions = await self._filter_transactions(
                transactions=transactions, user_context=user_context, **tool_args
            )

            # Calculate summary
            summary = self._calculate_summary(filtered_transactions)

            # Get category breakdown
            category_breakdown = self._get_category_breakdown(filtered_transactions)

            return {
                "transactions": filtered_transactions,
                "summary": summary,
                "category_breakdown": category_breakdown,
                "filters_applied": self._get_applied_filters(tool_args, user_context),
            }

        except Exception as e:
            logger.error(f"Error getting transactions: {e}")
            return {"error": f"Error retrieving transactions: {str(e)}"}

    async def create_transaction(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new transaction entry."""
        tool_args = context["tool_args"]
        user_context = context["user_context"]

        # Validate required fields
        amount = tool_args.get("amount", 0)
        description = tool_args.get("description", "").strip()

        if not description:
            return {"error": "Transaction description is required"}
        if amount == 0:
            return {"error": "Transaction amount cannot be zero"}

        # Find a target account
        account = await self._find_account(
            user_context=user_context,
            account_id=tool_args.get("account_id"),
            account_name=tool_args.get("account_name"),
        )

        if not account:
            return {"error": "Please specify a valid account for this transaction"}

        # Prepare transaction
        new_transaction = {
            "account_id": account["id"],
            "account_name": account["name"],
            "amount": float(amount),
            "description": description,
            "category": tool_args.get("category", "Uncategorized"),
            "date": self._get_transaction_date(tool_args.get("date")),
            "recurring": tool_args.get("recurring", False),
            "created_via_ai": True,
        }

        return {
            "action": "create_transaction",
            "transaction": new_transaction,
            "message": self._format_creation_message(new_transaction, account),
        }

    async def update_transaction(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing transaction."""
        tool_args = context["tool_args"]
        transaction_id = tool_args.get("transaction_id")

        if not transaction_id:
            return {"error": "Transaction ID is required for updates"}

        # Validate and prepare updates
        updates = await self._prepare_updates(tool_args)

        if "error" in updates:
            return updates

        if not updates:
            return {"error": "No valid updates provided"}

        return {
            "action": "update_transaction",
            "transaction_id": transaction_id,
            "updates": updates,
            "message": "I'll update the transaction with the changes you specified.",
        }

    @staticmethod
    async def delete_transaction(context: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a transaction."""
        tool_args = context["tool_args"]
        transaction_id = tool_args.get("transaction_id")

        if not transaction_id:
            return {"error": "Transaction ID is required for deletion"}

        return {
            "action": "delete_transaction",
            "transaction_id": transaction_id,
            "message": "I'll delete this transaction for you.",
            "requires_confirmation": True,
        }

    @staticmethod
    async def categorize_transactions(context: Dict[str, Any]) -> Dict[str, Any]:
        """Bulk categorize transactions."""
        transactions = context["transactions"]

        uncategorized = [
            t
            for t in transactions
            if not t.get("category") or t.get("category").lower() == "uncategorized"
        ]

        if not uncategorized:
            return {
                "message": "All transactions are already categorized!",
                "categorized_count": 0,
            }

        return {
            "action": "categorize_transactions",
            "transaction_count": len(uncategorized),
            "message": f"I'll categorize {len(uncategorized)} transactions for you using AI analysis.",
        }

    async def _filter_transactions(
        self,
        transactions: List[Dict[str, Any]],
        user_context: Dict[str, Any],
        **filters,
    ) -> List[Dict[str, Any]]:
        """Apply filters to transactions."""
        # Get date range
        today = datetime.now().date()
        days = filters.get("days", 30)
        start_date = filters.get("start_date")
        end_date = filters.get("end_date")

        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            start = today - timedelta(days=days)
            end = today

        # Get account filter
        target_account = None
        if filters.get("account_id") or filters.get("account_name"):
            target_account = await self._find_account(
                user_context=user_context,
                account_id=filters.get("account_id"),
                account_name=filters.get("account_name"),
            )

        # Apply filters
        filtered = []
        for txn in transactions:
            # Date filter
            txn_date = self._parse_transaction_date(txn["date"])
            if not (start <= txn_date <= end):
                continue

            # Account filter
            if target_account and str(txn.get("account_id")) != str(
                target_account["id"]
            ):
                continue

            # Category filter
            if (
                filters.get("category")
                and filters["category"].lower() not in txn.get("category", "").lower()
            ):
                continue

            # Amount filters
            amount = abs(float(txn.get("amount", 0)))
            if filters.get("min_amount") and amount < filters["min_amount"]:
                continue
            if filters.get("max_amount") and amount > filters["max_amount"]:
                continue

            # Type filter
            if filters.get("type") == "income" and float(txn["amount"]) <= 0:
                continue
            elif filters.get("type") == "expense" and float(txn["amount"]) >= 0:
                continue

            filtered.append(txn)

        # Sort by date (newest first)
        filtered.sort(key=lambda x: x["date"], reverse=True)
        return filtered

    @staticmethod
    async def _find_account(
        user_context: Dict[str, Any],
        account_id: Optional[str] = None,
        account_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find account by ID or name."""
        accounts = user_context.get("accounts", [])

        if account_id:
            return next(
                (acc for acc in accounts if str(acc.get("id")) == str(account_id)), None
            )

        if account_name:
            account_name_clean = account_name.strip().lower()
            # Try exact match first
            account = next(
                (
                    acc
                    for acc in accounts
                    if acc.get("name", "").lower() == account_name_clean
                ),
                None,
            )
            # Try partial match
            if not account:
                account = next(
                    (
                        acc
                        for acc in accounts
                        if account_name_clean in acc.get("name", "").lower()
                    ),
                    None,
                )
            return account

        return None

    async def _prepare_updates(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
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
            date = self._get_transaction_date(tool_args["date"])
            if not date:
                return {"error": "Invalid date format. Please use YYYY-MM-DD"}
            updates["date"] = date

        if "recurring" in tool_args:
            updates["recurring"] = bool(tool_args["recurring"])

        return updates

    @staticmethod
    def _calculate_summary(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate transaction summary."""
        total_income = sum(
            float(t["amount"]) for t in transactions if float(t["amount"]) > 0
        )
        total_expenses = sum(
            abs(float(t["amount"])) for t in transactions if float(t["amount"]) < 0
        )

        return {
            "count": len(transactions),
            "total_income": total_income,
            "total_expenses": total_expenses,
            "net_change": total_income - total_expenses,
        }

    @staticmethod
    def _get_category_breakdown(transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get spending by category."""
        breakdown = {}
        for txn in transactions:
            if float(txn["amount"]) < 0:  # Only expenses
                category = txn.get("category", "Uncategorized")
                breakdown[category] = breakdown.get(category, 0) + abs(
                    float(txn["amount"])
                )
        return breakdown

    @staticmethod
    def _get_transaction_date(date_str: Optional[str]) -> str:
        """Get a valid transaction date."""
        if date_str:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                return ""
        return datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def _parse_transaction_date(date_str: str) -> datetime.date:
        """Parse transaction date string."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            try:
                return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            except ValueError:
                return datetime.fromisoformat(date_str.split("T")[0]).date()

    @staticmethod
    def _format_creation_message(
        transaction: Dict[str, Any], account: Dict[str, Any]
    ) -> str:
        """Format transaction creation message."""
        amount = transaction["amount"]
        description = transaction["description"]
        account_name = account["name"]

        if amount > 0:
            return f"I'll create a ${amount:.2f} income transaction for '{description}' in your {account_name} account."
        else:
            return f"I'll create a ${abs(amount):.2f} expense transaction for '{description}' in your {account_name} account."

    @staticmethod
    def _get_applied_filters(
        tool_args: Dict[str, Any], user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get summary of applied filters."""
        accounts = user_context.get("accounts", [])
        return {
            "days": tool_args.get("days", 30),
            "category": tool_args.get("category"),
            "date_range": (
                f"{tool_args.get('start_date')} to {tool_args.get('end_date')}"
                if tool_args.get("start_date")
                else None
            ),
            "accounts_available": [acc.get("name") for acc in accounts],
        }

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def get_transactions(
    user_id: str,
    user_context: Dict[str, Any],
    transactions: List[Dict[str, Any]],
    **filters,
) -> Dict[str, Any]:
    """
    Retrieve and filter user transactions based on various criteria.

    Args:
        user_id: User identifier
        user_context: User profile data
        transactions: Full transaction list
        **filters: Various filter options (account_id, days, category, etc.)

    Returns:
        Dictionary containing filtered transactions and summary
    """
    try:
        # Parse filter parameters
        account_id = filters.get("account_id")
        account_name = filters.get("account_name", "").lower()
        days = filters.get("days", 30)
        category = filters.get("category", "").lower()
        start_date = filters.get("start_date")
        end_date = filters.get("end_date")
        min_amount = filters.get("min_amount")
        max_amount = filters.get("max_amount")
        transaction_type = filters.get("type", "").lower()  # 'income', 'expense', 'all'

        logger.info(f"🔍 Filtering transactions for user {user_id}")
        logger.info(f"🔍 Total transactions available: {len(transactions)}")
        logger.info(f"🔍 Account filter - ID: {account_id}, Name: '{account_name}'")
        logger.info(
            f"🔍 Date filter - Days: {days}, Start: {start_date}, End: {end_date}"
        )

        # Calculate date range
        today = datetime.now().date()
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            start = today - timedelta(days=days)
            end = today

        logger.info(f"🔍 Date range: {start} to {end}")

        # Get account info for filtering
        accounts = user_context.get("accounts", [])
        logger.info(f"🔍 Available accounts: {len(accounts)}")

        # Log account names for debugging
        for acc in accounts:
            logger.info(f"🔍 Account: ID={acc.get('id')}, Name='{acc.get('name')}'")

        target_account = None

        if account_id:
            target_account = next(
                (acc for acc in accounts if str(acc.get("id")) == str(account_id)), None
            )
            logger.info(f"🔍 Found account by ID: {target_account}")
        elif account_name:
            # More flexible account name matching
            account_name_clean = account_name.strip().lower()

            # Try exact match first
            target_account = next(
                (
                    acc
                    for acc in accounts
                    if acc.get("name", "").lower() == account_name_clean
                ),
                None,
            )

            # If no exact match, try partial match
            if not target_account:
                target_account = next(
                    (
                        acc
                        for acc in accounts
                        if account_name_clean in acc.get("name", "").lower()
                    ),
                    None,
                )

            # If still no match, try reverse partial match (account name contains filter)
            if not target_account:
                target_account = next(
                    (
                        acc
                        for acc in accounts
                        if acc.get("name", "").lower() in account_name_clean
                    ),
                    None,
                )

            logger.info(f"🔍 Found account by name '{account_name}': {target_account}")

        # Filter transactions
        filtered_transactions = []

        logger.info(f"🔍 Starting transaction filtering...")

        for i, txn in enumerate(transactions):
            try:
                # Log first few transactions for debugging
                if i < 3:
                    logger.info(
                        f"🔍 Transaction {i}: date={txn.get('date')}, account={txn.get('account')}, account_id={txn.get('account_id')}, amount={txn.get('amount')}"
                    )

                # Parse transaction date - handle multiple formats
                date_str = txn["date"]
                try:
                    # Try YYYY-MM-DD format first
                    txn_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    try:
                        # Try with timestamp format
                        txn_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                    except ValueError:
                        # Try parsing as ISO format
                        txn_date = datetime.fromisoformat(date_str.split("T")[0]).date()

                if i < 3:  # Log first few dates for debugging
                    logger.info(f"🔍 Transaction {i} date: '{date_str}' -> {txn_date}")

                # Date filter
                if not (start <= txn_date <= end):
                    continue

                # Account filter
                if target_account:
                    txn_account_id = txn.get("account_id")
                    target_account_id = target_account.get("id")

                    # Convert both to strings for comparison
                    if str(txn_account_id) != str(target_account_id):
                        if i < 5:  # Log first few mismatches
                            logger.info(
                                f"🔍 Transaction {i} account mismatch: txn_account_id='{txn_account_id}' vs target='{target_account_id}'"
                            )
                        continue

                # Category filter
                if category and category not in txn.get("category", "").lower():
                    continue

                # Amount filters
                amount = float(txn.get("amount", 0))
                if min_amount and abs(amount) < min_amount:
                    continue
                if max_amount and abs(amount) > max_amount:
                    continue

                # Transaction type filter
                if transaction_type == "income" and amount <= 0:
                    continue
                elif transaction_type == "expense" and amount >= 0:
                    continue

                filtered_transactions.append(txn)

            except Exception as e:
                logger.warning(f"Error processing transaction {i}: {e}")
                continue

        logger.info(
            f"🔍 Filtered transactions: {len(filtered_transactions)} out of {len(transactions)}"
        )

        # Sort by date (newest first)
        filtered_transactions.sort(key=lambda x: x["date"], reverse=True)

        # Calculate summary
        total_income = sum(
            float(t["amount"]) for t in filtered_transactions if float(t["amount"]) > 0
        )
        total_expenses = sum(
            abs(float(t["amount"]))
            for t in filtered_transactions
            if float(t["amount"]) < 0
        )
        net_change = total_income - total_expenses

        # Category breakdown
        category_spending = {}
        for txn in filtered_transactions:
            if float(txn["amount"]) < 0:  # Only expenses
                cat = txn.get("category", "Uncategorized")
                category_spending[cat] = category_spending.get(cat, 0) + abs(
                    float(txn["amount"])
                )

        result = {
            "transactions": filtered_transactions,
            "summary": {
                "count": len(filtered_transactions),
                "total_income": total_income,
                "total_expenses": total_expenses,
                "net_change": net_change,
                "date_range": f"{start} to {end}",
                "account": (
                    target_account.get("name") if target_account else "All accounts"
                ),
            },
            "category_breakdown": category_spending,
            "filters_applied": {
                "account": target_account.get("name") if target_account else None,
                "account_id": target_account.get("id") if target_account else None,
                "days": days,
                "category": category if category else None,
                "date_range": f"{start} to {end}",
                "total_available_transactions": len(transactions),
                "accounts_available": [acc.get("name") for acc in accounts],
            },
        }

        logger.info(
            f"🔍 Final result: {len(filtered_transactions)} transactions, account='{target_account.get('name') if target_account else 'All'}'"
        )

        return result

    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        import traceback

        traceback.print_exc()
        return {"error": f"Error retrieving transactions: {str(e)}"}


def create_transaction(
    user_id: str, user_context: Dict[str, Any], **transaction_data
) -> Dict[str, Any]:
    """
    Create a new transaction entry.

    Args:
        user_id: User identifier
        user_context: User profile data
        **transaction_data: Transaction details

    Returns:
        Dictionary with created transaction info or error
    """
    try:
        # Extract required fields
        amount = float(transaction_data.get("amount", 0))
        description = transaction_data.get("description", "").strip()
        account_id = transaction_data.get("account_id")
        account_name = transaction_data.get("account_name", "").lower()
        category = transaction_data.get("category", "Uncategorized")
        date_str = transaction_data.get("date")

        # Validate required fields
        if not description:
            return {"error": "Transaction description is required"}
        if amount == 0:
            return {"error": "Transaction amount cannot be zero"}

        # Find target account
        accounts = user_context.get("accounts", [])
        target_account = None

        if account_id:
            target_account = next(
                (acc for acc in accounts if str(acc.get("id")) == str(account_id)), None
            )
        elif account_name:
            target_account = next(
                (
                    acc
                    for acc in accounts
                    if account_name in acc.get("name", "").lower()
                ),
                None,
            )

        if not target_account:
            return {"error": "Please specify a valid account for this transaction"}

        # Parse date (default today)
        if date_str:
            try:
                transaction_date = datetime.strptime(date_str, "%Y-%m-%d").strftime(
                    "%Y-%m-%d"
                )
            except ValueError:
                return {"error": "Invalid date format. Please use YYYY-MM-DD"}
        else:
            transaction_date = datetime.now().strftime("%Y-%m-%d")

        # Prepare transaction data for creation
        new_transaction = {
            "account_id": target_account["id"],
            "account_name": target_account["name"],
            "amount": amount,
            "description": description,
            "category": category,
            "date": transaction_date,
            "recurring": transaction_data.get("recurring", False),
            "created_via_ai": True,
        }

        return {
            "action": "create_transaction",
            "transaction": new_transaction,
            "message": f"I'll create a {'${:.2f} income'.format(amount) if amount > 0 else '${:.2f} expense'.format(abs(amount))} transaction for '{description}' in your {target_account['name']} account.",
        }

    except Exception as e:
        logger.error(f"Error creating transaction: {e}")
        return {"error": f"Error creating transaction: {str(e)}"}


def update_transaction(user_id: str, transaction_id: str, **updates) -> Dict[str, Any]:
    """
    Update an existing transaction.

    Args:
        user_id: User identifier
        transaction_id: ID of transaction to update
        **updates: Fields to update

    Returns:
        Dictionary with update action or error
    """
    try:
        # Validate transaction_id
        if not transaction_id:
            return {"error": "Transaction ID is required for updates"}

        # Prepare update data
        update_data = {}

        if "amount" in updates:
            try:
                update_data["amount"] = float(updates["amount"])
                if update_data["amount"] == 0:
                    return {"error": "Transaction amount cannot be zero"}
            except (ValueError, TypeError):
                return {"error": "Invalid amount value"}

        if "description" in updates:
            desc = updates["description"].strip()
            if not desc:
                return {"error": "Description cannot be empty"}
            update_data["description"] = desc

        if "category" in updates:
            update_data["category"] = updates["category"].strip()

        if "date" in updates:
            try:
                datetime.strptime(updates["date"], "%Y-%m-%d")
                update_data["date"] = updates["date"]
            except ValueError:
                return {"error": "Invalid date format. Please use YYYY-MM-DD"}

        if "recurring" in updates:
            update_data["recurring"] = bool(updates["recurring"])

        if not update_data:
            return {"error": "No valid updates provided"}

        return {
            "action": "update_transaction",
            "transaction_id": transaction_id,
            "updates": update_data,
            "message": f"I'll update the transaction with the changes you specified.",
        }

    except Exception as e:
        logger.error(f"Error updating transaction: {e}")
        return {"error": f"Error updating transaction: {str(e)}"}


def delete_transaction(user_id: str, transaction_id: str) -> Dict[str, Any]:
    """
    Delete a transaction.

    Args:
        user_id: User identifier
        transaction_id: ID of transaction to delete

    Returns:
        Dictionary with delete action or error
    """
    try:
        if not transaction_id:
            return {"error": "Transaction ID is required for deletion"}

        return {
            "action": "delete_transaction",
            "transaction_id": transaction_id,
            "message": "I'll delete this transaction for you.",
            "requires_confirmation": True,
        }

    except Exception as e:
        logger.error(f"Error deleting transaction: {e}")
        return {"error": f"Error deleting transaction: {str(e)}"}


def categorize_transactions(
    user_id: str,
    transactions: List[Dict[str, Any]],
    category_mappings: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Bulk categorize transactions based on patterns or manual mappings.

    Args:
        user_id: User identifier
        transactions: List of transactions to categorize
        category_mappings: Optional manual category mappings

    Returns:
        Dictionary with categorization results
    """
    try:
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

        # This would integrate with your existing categorization service
        # For now, return an action for the backend to handle
        return {
            "action": "categorize_transactions",
            "transaction_count": len(uncategorized),
            "message": f"I'll categorize {len(uncategorized)} transactions for you using AI analysis.",
        }

    except Exception as e:
        logger.error(f"Error categorizing transactions: {e}")
        return {"error": f"Error categorizing transactions: {str(e)}"}

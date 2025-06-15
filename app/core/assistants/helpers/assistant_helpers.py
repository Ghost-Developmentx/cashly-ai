import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


async def enhance_query_with_context(
        query: str,
        user_context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]]
) -> str:
    """Add relevant context to the query."""
    if not user_context:
        return query

    context_parts = []

    # Add account summary if available
    if 'accounts' in user_context and user_context['accounts']:
        total_balance = sum(acc.get('balance', 0) for acc in user_context['accounts'])
        context_parts.append(
            f"User has {len(user_context['accounts'])} accounts with total balance ${total_balance:,.2f}")

    # Add recent activity summary if transactions provided
    if 'transactions' in user_context and len(user_context['transactions']) > 0:
        context_parts.append(f"User has {len(user_context['transactions'])} recent transactions")

    if context_parts:
        context_str = "Context: " + "; ".join(context_parts) + "\n\n"
        return context_str + query

    return query
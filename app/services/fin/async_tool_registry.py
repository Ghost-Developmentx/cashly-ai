"""
Async tool registry for financial operations.
Dispatches tool calls and exposes tool schemas.
"""

import logging
from typing import Any, Dict, List, Optional

from .tool_schemas import TOOL_SCHEMAS
from .async_tool_executor import AsyncToolExecutor
from .async_rails_client import AsyncRailsClient
from .tool_handlers import (
    AsyncAccountHandlers,
    AsyncTransactionHandlers,
    AsyncInvoiceHandlers,
    AsyncStripeHandlers,
    AsyncAnalyticsHandlers,
)

logger = logging.getLogger(__name__)


class AsyncToolRegistry:
    """
    Registry for asynchronous tools and their handlers.

    This class manages the registration and execution of various asynchronous tool
    handlers, such as account tools, transaction tools, invoice tools, Stripe
    Connect tools, and analytics tools. It provides a centralized interface to
    execute tools, manage their schemas, and handle related resources. The tools
    are executed asynchronously, and the registry is designed to support scalable
    and modular addition of different handlers.

    Attributes
    ----------
    rails_client : AsyncRailsClient
        Client for async interaction with Rails-based services.
    executor : AsyncToolExecutor
        Executor responsible for running the tools asynchronously.
    account_handlers : AsyncAccountHandlers
        Handlers for operations related to user accounts.
    transaction_handlers : AsyncTransactionHandlers
        Handlers for operations related to transactions.
    invoice_handlers : AsyncInvoiceHandlers
        Handlers for operations related to invoices.
    stripe_handlers : AsyncStripeHandlers
        Handlers for operations related to Stripe Connect.
    analytics_handlers : AsyncAnalyticsHandlers
        Handlers for analytics-related operations.
    _handlers : dict
        Registry-mapping tool names to their respective handler functions.
    """

    def __init__(self):
        # Initialize async clients and services
        self.rails_client = AsyncRailsClient()
        self.executor = AsyncToolExecutor()

        # Initialize handlers
        self.account_handlers = AsyncAccountHandlers(self.rails_client)
        self.transaction_handlers = AsyncTransactionHandlers()
        self.invoice_handlers = AsyncInvoiceHandlers(self.rails_client)
        self.stripe_handlers = AsyncStripeHandlers(self.rails_client)
        self.analytics_handlers = AsyncAnalyticsHandlers()

        # Register all handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register all tool handlers."""
        self._handlers = {
            # Account tools
            "get_user_accounts": self.account_handlers.get_user_accounts,
            "get_account_details": self.account_handlers.get_account_details,
            "initiate_plaid_connection": self.account_handlers.initiate_plaid_connection,
            "disconnect_account": self.account_handlers.disconnect_account,
            # Transaction tools
            "get_transactions": self.transaction_handlers.get_transactions,
            "create_transaction": self.transaction_handlers.create_transaction,
            "update_transaction": self.transaction_handlers.update_transaction,
            "delete_transaction": self.transaction_handlers.delete_transaction,
            "categorize_transactions": self.transaction_handlers.categorize_transactions,
            # Invoice tools
            "connect_stripe": self.invoice_handlers.connect_stripe,
            "create_invoice": self.invoice_handlers.create_invoice,
            "send_invoice": self.invoice_handlers.send_invoice,
            "delete_invoice": self.invoice_handlers.delete_invoice,
            "get_invoices": self.invoice_handlers.get_invoices,
            "send_invoice_reminder": self.invoice_handlers.send_invoice_reminder,
            "mark_invoice_paid": self.invoice_handlers.mark_invoice_paid,
            # Stripe Connect tools
            "setup_stripe_connect": self.stripe_handlers.setup_stripe_connect,
            "check_stripe_connect_status": self.stripe_handlers.check_stripe_connect_status,
            "create_stripe_connect_dashboard_link": self.stripe_handlers.create_stripe_connect_dashboard_link,
            "get_stripe_connect_earnings": self.stripe_handlers.get_stripe_connect_earnings,
            "disconnect_stripe_connect": self.stripe_handlers.disconnect_stripe_connect,
            "restart_stripe_connect_setup": self.stripe_handlers.restart_stripe_connect_setup,
            "resume_stripe_connect_onboarding": self.stripe_handlers.resume_stripe_connect_onboarding,
            "get_stripe_connect_requirements": self.stripe_handlers.get_stripe_connect_requirements,
            # Analytics tools
            "forecast_cash_flow": self.analytics_handlers.forecast_cash_flow,
            "analyze_trends": self.analytics_handlers.analyze_trends,
            "detect_anomalies": self.analytics_handlers.detect_anomalies,
            "generate_budget": self.analytics_handlers.generate_budget,
            "calculate_category_spending": self.analytics_handlers.calculate_category_spending,
        }

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas."""
        return TOOL_SCHEMAS

    async def execute(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        *,
        user_id: str,
        transactions: List[Dict[str, Any]],
        user_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a tool asynchronously.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Tool arguments
            user_id: User identifier
            transactions: User transactions
            user_context: User context data

        Returns:
            Tool execution result
        """
        handler = self._handlers.get(tool_name)
        if not handler:
            logger.warning(f"Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

        # Prepare execution context
        context = {
            "user_id": user_id,
            "transactions": transactions,
            "user_context": user_context,
            "tool_args": tool_args,
        }

        try:
            # Execute tool asynchronously
            result = await self.executor.execute_tool(handler, context)
            return result
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}

    async def close(self):
        """Close async resources."""
        await self.rails_client.close()

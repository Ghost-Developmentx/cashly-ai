from typing import Any, Dict, List
import logging

from services.forecast_service import ForecastService
from services.categorize_service import CategorizationService
from services.budget_service import BudgetService
from services.insight_service import InsightService
from services.anomaly_service import AnomalyService

from services.fin.tool_schemas import TOOL_SCHEMAS
from services.fin.utils import normalize_transaction_dates
from services.fin.account_helpers import (
    get_user_accounts,
    get_account_details,
    initiate_plaid_connection,
    disconnect_account,
)
from services.fin.transaction_helpers import (
    get_transactions,
    create_transaction,
    update_transaction,
    delete_transaction,
    categorize_transactions,
)
from services.fin.invoice_helpers import (
    connect_stripe_account,
    create_invoice,
    get_invoices,
    send_invoice_reminder,
    mark_invoice_paid,
)
from services.fin.stripe_connect_helpers import (
    setup_stripe_connect,
    check_stripe_connect_status,
    create_stripe_connect_dashboard_link,
    get_stripe_connect_earnings,
    disconnect_stripe_connect,
)


logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Responsible for dispatching tool calls and exposing tool schemas.
    """

    def __init__(self):
        self.forecast_service = ForecastService()
        self.categorization_service = CategorizationService()
        self.budget_service = BudgetService()
        self.insight_service = InsightService()
        self.anomaly_service = AnomalyService()

        self._handlers = {
            "forecast_cash_flow": self._forecast_cash_flow,
            "analyze_trends": self._analyze_trends,
            "detect_anomalies": self._detect_anomalies,
            "generate_budget": self._generate_budget,
            "calculate_category_spending": self._calculate_category_spending,
            "get_user_accounts": self._get_user_accounts,
            "get_account_details": self._get_account_details,
            "initiate_plaid_connection": self._initiate_plaid_connection,
            "disconnect_account": self._disconnect_account,
            "get_transactions": self._get_transactions,
            "create_transaction": self._create_transaction,
            "update_transaction": self._update_transaction,
            "delete_transaction": self._delete_transaction,
            "categorize_transactions": self._categorize_transactions,
            "connect_stripe": self._connect_stripe,
            "create_invoice": self._create_invoice,
            "get_invoices": self._get_invoices,
            "send_invoice_reminder": self._send_invoice_reminder,
            "mark_invoice_paid": self._mark_invoice_paid,
            "setup_stripe_connect": self._setup_stripe_connect,
            "check_stripe_connect_status": self._check_stripe_connect_status,
            "create_stripe_connect_dashboard_link": self._create_stripe_connect_dashboard_link,
            "get_stripe_connect_earnings": self._get_stripe_connect_earnings,
            "disconnect_stripe_connect": self._disconnect_stripe_connect,
        }

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    def execute(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        *,
        user_id: str,
        transactions: List[Dict[str, Any]],
        user_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        handler = self._handlers.get(tool_name)
        if not handler:
            logger.warning(f"Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

        txns = normalize_transaction_dates(transactions)
        return handler(tool_args, user_id=user_id, txns=txns, user_context=user_context)

    # ---------------------------------------------------------------------
    # Tool Handlers (delegating to services)
    # ---------------------------------------------------------------------

    def _forecast_cash_flow(self, args, *, user_id, txns, _user_context):
        days = args.get("days", 30)
        adjustments = args.get("adjustments", {})
        if adjustments:
            return self.forecast_service.forecast_cash_flow_scenario(
                user_id=user_id,
                transactions=txns,
                forecast_days=days,
                adjustments=adjustments,
            )
        return self.forecast_service.forecast_cash_flow(
            user_id=user_id, transactions=txns, forecast_days=days
        )

    def _analyze_trends(self, args, *, user_id, txns, _user_context):
        return self.insight_service.analyze_trends(
            user_id=user_id, transactions=txns, period=args.get("period", "3m")
        )

    def _detect_anomalies(self, args, *, user_id, txns, _user_context):
        return self.anomaly_service.detect_anomalies(
            user_id=user_id, transactions=txns, threshold=args.get("threshold")
        )

    def _generate_budget(self, args, *, user_id, txns, _user_context):
        return self.budget_service.generate_budget(
            user_id=user_id,
            transactions=txns,
            monthly_income=args.get("monthly_income"),
        )

    @staticmethod
    def _calculate_category_spending(args, *, _user_id, txns, _user_context):
        from datetime import datetime, timedelta

        categories = args.get("categories", [])
        start_date_str = args.get("start_date")
        end_date_str = args.get("end_date")

        today = datetime.now().date()
        start_date = (
            datetime.strptime(start_date_str, "%Y-%m-%d").date()
            if start_date_str
            else today - timedelta(days=30)
        )
        end_date = (
            datetime.strptime(end_date_str, "%Y-%m-%d").date()
            if end_date_str
            else today
        )

        filtered_txns = [
            t
            for t in txns
            if start_date <= datetime.strptime(t["date"], "%Y-%m-%d").date() <= end_date
        ]

        if categories:
            filtered_txns = [
                t
                for t in filtered_txns
                if any(
                    c.lower() in str(t.get("category", "")).lower() for c in categories
                )
            ]

        totals = {}
        for t in filtered_txns:
            if t["amount"] < 0:
                category = t.get("category", "Uncategorized")
                totals[category] = totals.get(category, 0) + abs(t["amount"])

        return {
            "time_period": f"{start_date} to {end_date}",
            "total_spending": sum(totals.values()),
            "category_breakdown": totals,
            "transaction_count": len(filtered_txns),
        }

    @staticmethod
    def _get_user_accounts(_args, *, user_id, txns, user_context):
        return get_user_accounts(user_id, user_context)

    @staticmethod
    def _get_account_details(args, *, user_id, txns, user_context):
        return get_account_details(user_id, args.get("account_id"), user_context)

    @staticmethod
    def _initiate_plaid_connection(args, *, user_id, txns, user_context):
        return initiate_plaid_connection(user_id, args.get("institution_preference"))

    @staticmethod
    def _disconnect_account(args, *, user_id, txns, user_context):
        return disconnect_account(user_id, args.get("account_id"))

    @staticmethod
    def _get_transactions(args, *, user_id, txns, user_context):
        return get_transactions(user_id, user_context, txns, **args)

    @staticmethod
    def _create_transaction(args, *, user_id, txns, user_context):
        return create_transaction(user_id, user_context, **args)

    @staticmethod
    def _update_transaction(args, *, user_id, txns, user_context):
        transaction_id = args.pop("transaction_id", None)
        return update_transaction(user_id, transaction_id, **args)

    @staticmethod
    def _delete_transaction(args, *, user_id, txns, user_context):
        return delete_transaction(user_id, args.get("transaction_id"))

    @staticmethod
    def _categorize_transactions(args, *, user_id, txns, user_context):
        return categorize_transactions(user_id, txns, args.get("category_mappings"))

    @staticmethod
    def _connect_stripe(args, *, user_id, txns, user_context):
        return connect_stripe_account(user_id, args.get("api_key"))

    @staticmethod
    def _create_invoice(args, *, user_id, txns, user_context):
        return create_invoice(user_id, args)

    @staticmethod
    def _get_invoices(args, *, user_id, txns, user_context):
        return get_invoices(user_id, **args)

    @staticmethod
    def _send_invoice_reminder(args, *, user_id, txns, user_context):
        return send_invoice_reminder(user_id, args.get("invoice_id"))

    @staticmethod
    def _mark_invoice_paid(args, *, user_id, txns, user_context):
        return mark_invoice_paid(user_id, args.get("invoice_id"))

    @staticmethod
    def _setup_stripe_connect(args, *, user_id, txns, user_context):
        return setup_stripe_connect(
            user_id,
            user_context,
            country=args.get("country", "US"),
            business_type=args.get("business_type", "individual"),
        )

    @staticmethod
    def _check_stripe_connect_status(args, *, user_id, txns, user_context):
        return check_stripe_connect_status(user_id, user_context)

    @staticmethod
    def _create_stripe_connect_dashboard_link(args, *, user_id, txns, user_context):
        return create_stripe_connect_dashboard_link(user_id, user_context)

    @staticmethod
    def _get_stripe_connect_earnings(args, *, user_id, txns, user_context):
        return get_stripe_connect_earnings(
            user_id, user_context, period=args.get("period", "month")
        )

    @staticmethod
    def _disconnect_stripe_connect(args, *, user_id, txns, user_context):
        return disconnect_stripe_connect(user_id, user_context)

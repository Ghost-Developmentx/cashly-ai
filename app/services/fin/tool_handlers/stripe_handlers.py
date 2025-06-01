"""
Async handlers for Stripe Connect tools.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class AsyncStripeHandlers:
    """
    Handles asynchronous operations related to Stripe Connect integration.

    This class provides a variety of methods to manage and interact with Stripe Connect
    accounts for users. It includes functionalities for setting up accounts,
    checking account statuses, creating dashboard links, managing earnings, and
    handling disconnections or onboarding processes. The class is designed to work
    with an external `rails_client` for communication and specifically tailored for
    asynchronous use cases.

    Attributes
    ----------
    rails_client
        The client instance used for communication with an external service or API.
    """

    def __init__(self, rails_client):
        self.rails_client = rails_client

    async def setup_stripe_connect(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Set up Stripe Connect for user."""
        tool_args = context["tool_args"]
        user_context = context["user_context"]
        user_id = context["user_id"]

        stripe_status = user_context.get("stripe_connect", {})

        # Check if already connected
        if stripe_status.get("connected") and stripe_status.get("can_accept_payments"):
            return {
                "message": "Your Stripe Connect account is already set up and working!",
                "already_connected": True,
                "status": stripe_status.get("status"),
                "can_accept_payments": True,
            }

        # Handle incomplete or rejected accounts
        if stripe_status.get("connected"):
            return await self._handle_existing_account(user_id, stripe_status)

        # Fresh setup
        return {
            "action": "setup_stripe_connect",
            "user_id": user_id,
            "country": tool_args.get("country", "US"),
            "business_type": tool_args.get("business_type", "individual"),
            "message": "I'll help you set up Stripe Connect so you can accept payments.",
            "benefits": [
                "Accept credit card and bank payments",
                "Professional invoice presentation",
                "Automatic payment tracking",
                "Stripe's fraud protection",
            ],
            "platform_fee": "2.9% + Stripe's processing fees",
        }

    async def check_stripe_connect_status(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check Stripe Connect account status."""
        user_context = context["user_context"]
        stripe_status = user_context.get("stripe_connect", {})

        if not stripe_status.get("connected"):
            return {
                "connected": False,
                "message": "You don't have Stripe Connect set up yet.",
                "setup_recommended": True,
            }

        status_info = self._analyze_stripe_status(stripe_status)

        return {
            "connected": True,
            **status_info,
            "message": self._get_status_message(status_info),
        }

    @staticmethod
    async def create_stripe_connect_dashboard_link(
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create Stripe dashboard link."""
        user_context = context["user_context"]
        stripe_status = user_context.get("stripe_connect", {})

        if not stripe_status.get("connected"):
            return {"error": "You need to set up Stripe Connect first."}

        return {
            "action": "create_stripe_connect_dashboard_link",
            "user_id": context["user_id"],
            "message": "I'll open your Stripe dashboard where you can manage your account.",
            "note": "The dashboard will show any pending requirements.",
        }

    @staticmethod
    async def get_stripe_connect_earnings(context: Dict[str, Any]) -> Dict[str, Any]:
        """Get Stripe Connect earnings."""
        tool_args = context["tool_args"]
        user_context = context["user_context"]
        stripe_status = user_context.get("stripe_connect", {})

        if not stripe_status.get("connected"):
            return {
                "error": "You need to set up Stripe Connect first to view earnings."
            }

        return {
            "action": "get_stripe_connect_earnings",
            "period": tool_args.get("period", "month"),
            "user_id": context["user_id"],
            "message": f"I'll get your Stripe Connect earnings for the {tool_args.get('period', 'month')}.",
        }

    @staticmethod
    async def disconnect_stripe_connect(context: Dict[str, Any]) -> Dict[str, Any]:
        """Disconnect Stripe Connect account."""
        user_context = context["user_context"]
        stripe_status = user_context.get("stripe_connect", {})

        if not stripe_status.get("connected"):
            return {"error": "You don't have a Stripe Connect account to disconnect."}

        return {
            "action": "disconnect_stripe_connect",
            "user_id": context["user_id"],
            "message": "I'll disconnect your Stripe Connect account.",
            "requires_confirmation": True,
            "warning": "You won't be able to accept payments until you reconnect.",
        }

    @staticmethod
    async def restart_stripe_connect_setup(context: Dict[str, Any]) -> Dict[str, Any]:
        """Restart Stripe Connect setup."""
        return {
            "action": "restart_stripe_connect",
            "user_id": context["user_id"],
            "message": "I'll help you start the Stripe Connect setup from the beginning.",
            "warning": "This will disconnect any existing Stripe Connect account.",
            "requires_confirmation": True,
        }

    async def resume_stripe_connect_onboarding(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resume Stripe Connect onboarding."""
        tool_args = context["tool_args"]
        action = tool_args.get("action", "continue")

        if action == "restart":
            return await self.restart_stripe_connect_setup(context)

        return {
            "action": "create_stripe_connect_dashboard_link",
            "user_id": context["user_id"],
            "message": "I'll help you continue your Stripe Connect setup.",
        }

    async def get_stripe_connect_requirements(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get Stripe Connect requirements."""
        user_context = context["user_context"]
        stripe_status = user_context.get("stripe_connect", {})

        if not stripe_status.get("connected"):
            return {
                "message": "You don't have a Stripe Connect account yet.",
                "action": "setup_stripe_connect",
            }

        requirements = stripe_status.get("requirements", {})

        return {
            "current_status": stripe_status.get("status"),
            "requirements": requirements,
            "message": f"Your Stripe Connect account status is: {stripe_status.get('status')}",
            "next_steps": self._get_next_steps(stripe_status),
            "can_accept_payments": stripe_status.get("can_accept_payments", False),
        }

    @staticmethod
    async def _handle_existing_account(
        user_id: str, stripe_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle existing incomplete/rejected accounts."""
        status = stripe_status.get("status", "unknown")

        if status == "rejected":
            return {
                "action": "resume_stripe_connect_after_rejection",
                "user_id": user_id,
                "message": "Your previous Stripe setup encountered issues. Let me help you resolve them.",
                "options": [
                    {
                        "action": "create_new_stripe_account",
                        "text": "Start Fresh with New Account",
                    },
                    {
                        "action": "resume_existing_stripe_account",
                        "text": "Fix Existing Account",
                    },
                ],
                "current_status": stripe_status,
            }

        elif not stripe_status.get("onboarding_complete"):
            return {
                "action": "resume_stripe_connect_onboarding",
                "user_id": user_id,
                "message": "You have an incomplete Stripe Connect setup. Let's finish it.",
                "options": [
                    {
                        "action": "continue_onboarding",
                        "text": "Continue Setup",
                    },
                    {
                        "action": "start_over",
                        "text": "Start Over",
                    },
                ],
                "current_status": stripe_status,
            }

        return {
            "message": "Your Stripe Connect account needs attention.",
            "status": status,
        }

    @staticmethod
    def _analyze_stripe_status(stripe_status: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Stripe account status."""
        return {
            "status": stripe_status.get("status", "unknown"),
            "charges_enabled": stripe_status.get("charges_enabled", False),
            "payouts_enabled": stripe_status.get("payouts_enabled", False),
            "onboarding_complete": stripe_status.get("onboarding_complete", False),
            "can_accept_payments": (
                stripe_status.get("charges_enabled", False)
                and stripe_status.get("status") == "active"
            ),
            "platform_fee_percentage": stripe_status.get(
                "platform_fee_percentage", 2.9
            ),
            "requirements": stripe_status.get("requirements", {}),
        }

    @staticmethod
    def _get_status_message(status_info: Dict[str, Any]) -> str:
        """Get appropriate status message."""
        status = status_info["status"]
        charges_enabled = status_info["charges_enabled"]

        if status == "active" and charges_enabled:
            return "Your Stripe Connect account is active and ready! 🎉"
        elif status == "pending":
            if status_info["onboarding_complete"]:
                return "Your account is pending review. You should be able to accept payments soon."
            else:
                return "Please complete your Stripe Connect setup to start accepting payments."
        elif status == "rejected":
            return "Your account was rejected. Please contact support for assistance."
        else:
            return f"Your Stripe Connect account status is: {status}"

    @staticmethod
    def _get_next_steps(stripe_status: Dict[str, Any]) -> List[str]:
        """Get next steps based on status."""
        if not stripe_status.get("onboarding_complete"):
            return [
                "Complete the onboarding process",
                "Provide all required information",
                "Verify your identity",
            ]

        if stripe_status.get("requirements"):
            return [
                "Complete pending requirements in Stripe dashboard",
                "Verify your business information",
                "Submit required documents",
            ]

        return ["Your account is set up and ready to use!"]

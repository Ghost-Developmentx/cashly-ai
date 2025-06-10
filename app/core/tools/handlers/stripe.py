"""
Stripe Connect tool handlers.
Migrated from app/services/fin/tool_handlers/stripe_handlers.py
"""

import logging
from typing import Dict, Any
from ..registry import tool_registry
from ..schemas import STRIPE_SCHEMAS

logger = logging.getLogger(__name__)

@tool_registry.register(
    name="setup_stripe_connect",
    description="Setup Stripe Connect for accepting payments",
    schema=STRIPE_SCHEMAS["SETUP_STRIPE_CONNECT"],
    category="stripe"
)
async def setup_stripe_connect(context: Dict[str, Any]) -> Dict[str, Any]:
    """Setup Stripe Connect account."""
    user_id = context.get("user_id")
    user_context = context.get("user_context", {})
    stripe_connect = user_context.get("stripe_connect", {})

    # Check if already connected
    if stripe_connect.get("connected") and stripe_connect.get("can_accept_payments"):
        return {
            "already_connected": True,
            "status": stripe_connect.get("status", "active"),
            "can_accept_payments": True,
            "message": "Your Stripe Connect account is already set up and ready to accept payments!"
        }

    # Check if onboarding in progress
    if stripe_connect.get("onboarding_started") and not stripe_connect.get("connected"):
        return {
            "action": "resume_stripe_connect_onboarding",
            "user_id": user_id,
            "message": "You have an incomplete Stripe Connect setup. Let's continue where you left off."
        }

    return {
        "action": "setup_stripe_connect",
        "user_id": user_id,
        "message": "Let's set up your Stripe Connect account so you can accept payments directly.",
        "benefits": [
            "Accept payments directly to your bank account",
            "Automatic payment processing",
            "Professional invoicing",
            "Secure and PCI compliant"
        ]
    }

@tool_registry.register(
    name="check_stripe_connect_status",
    description="Check the status of Stripe Connect account",
    schema=STRIPE_SCHEMAS["CHECK_STRIPE_CONNECT_STATUS"],
    category="stripe"
)
async def check_stripe_connect_status(context: Dict[str, Any]) -> Dict[str, Any]:
    """Check Stripe Connect account status."""
    user_context = context.get("user_context", {})
    stripe_connect = user_context.get("stripe_connect", {})

    if not stripe_connect or not stripe_connect.get("connected"):
        return {
            "connected": False,
            "status": "not_connected",
            "message": "You haven't connected a Stripe account yet.",
            "setup_recommended": True
        }

    status = stripe_connect.get("status", "unknown")
    charges_enabled = stripe_connect.get("charges_enabled", False)
    payouts_enabled = stripe_connect.get("payouts_enabled", False)

    # Build status message
    if status == "active" and charges_enabled:
        status_message = "Your Stripe Connect account is active and can accept payments."
    elif status == "restricted":
        status_message = "Your account has restrictions. Additional information may be required."
    elif status == "pending":
        status_message = "Your account is pending verification."
    else:
        status_message = f"Your account status is: {status}"

    return {
        "connected": True,
        "status": status,
        "charges_enabled": charges_enabled,
        "payouts_enabled": payouts_enabled,
        "can_accept_payments": charges_enabled,
        "message": status_message,
        "details": {
            "account_id": stripe_connect.get("account_id"),
            "business_type": stripe_connect.get("business_type"),
            "country": stripe_connect.get("country", "US"),
            "created_at": stripe_connect.get("created_at")
        }
    }

@tool_registry.register(
    name="create_stripe_connect_dashboard_link",
    description="Create a link to the Stripe Connect dashboard",
    schema=STRIPE_SCHEMAS["CREATE_STRIPE_CONNECT_DASHBOARD_LINK"],
    category="stripe"
)
async def create_stripe_connect_dashboard_link(context: Dict[str, Any]) -> Dict[str, Any]:
    """Create Stripe Connect dashboard link."""
    user_id = context.get("user_id")
    user_context = context.get("user_context", {})
    stripe_connect = user_context.get("stripe_connect", {})

    if not stripe_connect.get("connected"):
        return {
            "error": "No Stripe Connect account found. Please set up Stripe Connect first."
        }

    return {
        "action": "create_stripe_connect_dashboard_link",
        "user_id": user_id,
        "message": "I'll create a secure link to your Stripe dashboard where you can manage payments, view reports, and update settings."
    }

@tool_registry.register(
    name="get_stripe_connect_earnings",
    description="Get earnings from Stripe Connect",
    schema=STRIPE_SCHEMAS["GET_STRIPE_CONNECT_EARNINGS"],
    category="stripe"
)
async def get_stripe_connect_earnings(context: Dict[str, Any]) -> Dict[str, Any]:
    """Get Stripe Connect earnings summary."""
    tool_args = context["tool_args"]
    user_context = context.get("user_context", {})
    stripe_connect = user_context.get("stripe_connect", {})

    if not stripe_connect.get("connected"):
        return {
            "error": "No Stripe Connect account found. Please set up Stripe Connect first."
        }

    period = tool_args.get("period", "month")

    # In a real implementation, this would fetch from Stripe API
    # For now, return action for frontend
    return {
        "action": "get_stripe_connect_earnings",
        "period": period,
        "message": f"I'll fetch your earnings for the {period}.",
        "stripe_account_id": stripe_connect.get("account_id")
    }

@tool_registry.register(
    name="disconnect_stripe_connect",
    description="Disconnect Stripe Connect account",
    schema=STRIPE_SCHEMAS["DISCONNECT_STRIPE_CONNECT"],
    category="stripe",
    requires_confirmation=True
)
async def disconnect_stripe_connect(context: Dict[str, Any]) -> Dict[str, Any]:
    """Disconnect Stripe Connect account."""
    user_id = context.get("user_id")
    user_context = context.get("user_context", {})
    stripe_connect = user_context.get("stripe_connect", {})

    if not stripe_connect.get("connected"):
        return {
            "error": "No Stripe Connect account to disconnect."
        }

    return {
        "action": "disconnect_stripe_connect",
        "user_id": user_id,
        "message": "Are you sure you want to disconnect your Stripe account? You won't be able to accept payments until you reconnect.",
        "requires_confirmation": True,
        "warning": "Active invoices and payment links will stop working."
    }

@tool_registry.register(
    name="restart_stripe_connect_setup",
    description="Restart Stripe Connect setup process",
    schema=STRIPE_SCHEMAS["RESTART_STRIPE_CONNECT_SETUP"],
    category="stripe"
)
async def restart_stripe_connect_setup(context: Dict[str, Any]) -> Dict[str, Any]:
    """Restart Stripe Connect setup."""
    user_id = context.get("user_id")

    return {
        "action": "restart_stripe_connect_setup",
        "user_id": user_id,
        "message": "I'll help you start fresh with a new Stripe Connect setup.",
        "note": "This will create a new onboarding session."
    }

@tool_registry.register(
    name="resume_stripe_connect_onboarding",
    description="Resume incomplete Stripe Connect onboarding",
    schema=STRIPE_SCHEMAS["RESUME_STRIPE_CONNECT_ONBOARDING"],
    category="stripe"
)
async def resume_stripe_connect_onboarding(context: Dict[str, Any]) -> Dict[str, Any]:
    """Resume Stripe Connect onboarding."""
    user_id = context.get("user_id")
    user_context = context.get("user_context", {})
    stripe_connect = user_context.get("stripe_connect", {})

    if stripe_connect.get("connected"):
        return {
            "message": "Your Stripe Connect account is already connected!",
            "connected": True
        }

    if not stripe_connect.get("onboarding_started"):
        return {
            "message": "No onboarding session found. Would you like to start setting up Stripe Connect?",
            "suggest_action": "setup_stripe_connect"
        }

    return {
        "action": "resume_stripe_connect_onboarding",
        "user_id": user_id,
        "message": "Let's continue setting up your Stripe Connect account where you left off."
    }

@tool_registry.register(
    name="get_stripe_connect_requirements",
    description="Get requirements for Stripe Connect verification",
    schema=STRIPE_SCHEMAS["GET_STRIPE_CONNECT_REQUIREMENTS"],
    category="stripe"
)
async def get_stripe_connect_requirements(context: Dict[str, Any]) -> Dict[str, Any]:
    """Get Stripe Connect requirements."""
    user_context = context.get("user_context", {})
    stripe_connect = user_context.get("stripe_connect", {})

    if not stripe_connect.get("connected"):
        return {
            "error": "No Stripe Connect account found. Please set up Stripe Connect first."
        }

    # Check for any requirements
    requirements = stripe_connect.get("requirements", {})
    currently_due = requirements.get("currently_due", [])
    eventually_due = requirements.get("eventually_due", [])

    if not currently_due and not eventually_due:
        return {
            "message": "Great! Your Stripe Connect account has no outstanding requirements.",
            "all_requirements_met": True
        }

    return {
        "currently_due": currently_due,
        "eventually_due": eventually_due,
        "message": f"Your Stripe account needs {len(currently_due)} item{'s' if len(currently_due) != 1 else ''} to be completed.",
        "action_required": len(currently_due) > 0
    }

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def setup_stripe_connect(
    user_id: str,
    user_context: Dict[str, Any],
    country: str = "US",
    business_type: str = "individual",
) -> Dict[str, Any]:
    """
    Initiate Stripe Connect setup for the user.

    Args:
        user_id: User identifier
        user_context: User profile data
        country: Country code for the business
        business_type: Type of business entity

    Returns:
        Dictionary with setup action or error
    """
    try:
        # Check if user already has Stripe Connect
        stripe_status = user_context.get("stripe_connect", {})

        if stripe_status.get("connected"):
            return {
                "message": "You already have Stripe Connect set up! You can manage your payments through the Stripe dashboard.",
                "already_connected": True,
                "status": stripe_status.get("status"),
                "can_accept_payments": stripe_status.get("can_accept_payments", False),
            }

        return {
            "action": "setup_stripe_connect",
            "user_id": user_id,
            "country": country,
            "business_type": business_type,
            "message": "I'll help you set up Stripe Connect so you can accept payments directly through your invoices. This will enable you to send professional invoices and receive payments with a small platform fee.",
            "benefits": [
                "Accept credit card and bank payments",
                "Professional invoice presentation",
                "Automatic payment tracking",
                "Stripe's fraud protection",
                "Instant payment notifications",
            ],
            "platform_fee": "2.9% + Stripe's processing fees",
        }

    except Exception as e:
        logger.error(f"Error setting up Stripe Connect: {e}")
        return {"error": f"Error setting up Stripe Connect: {str(e)}"}


def check_stripe_connect_status(
    user_id: str, user_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check the current Stripe Connect status for the user.

    Args:
        user_id: User identifier
        user_context: User profile data

    Returns:
        Dictionary with current status information
    """
    try:
        stripe_status = user_context.get("stripe_connect", {})

        if not stripe_status.get("connected"):
            return {
                "connected": False,
                "message": "You don't have Stripe Connect set up yet. Would you like me to help you set it up so you can accept payments through your invoices?",
                "setup_recommended": True,
            }

        status = stripe_status.get("status", "unknown")
        charges_enabled = stripe_status.get("charges_enabled", False)
        payouts_enabled = stripe_status.get("payouts_enabled", False)
        onboarding_complete = stripe_status.get("onboarding_complete", False)

        if status == "active" and charges_enabled:
            message = (
                "Your Stripe Connect account is active and ready to accept payments! 🎉"
            )
        elif status == "pending":
            if onboarding_complete:
                message = "Your Stripe Connect account is pending review. You should be able to accept payments soon."
            else:
                message = "Your Stripe Connect setup is incomplete. Please finish the onboarding process to start accepting payments."
        elif status == "rejected":
            message = "Your Stripe Connect account was rejected. Please contact support for assistance."
        else:
            message = f"Your Stripe Connect account status is: {status}"

        return {
            "connected": True,
            "status": status,
            "charges_enabled": charges_enabled,
            "payouts_enabled": payouts_enabled,
            "onboarding_complete": onboarding_complete,
            "can_accept_payments": charges_enabled and status == "active",
            "platform_fee_percentage": stripe_status.get(
                "platform_fee_percentage", 2.9
            ),
            "message": message,
            "requirements": stripe_status.get("requirements", {}),
        }

    except Exception as e:
        logger.error(f"Error checking Stripe Connect status: {e}")
        return {"error": f"Error checking Stripe Connect status: {str(e)}"}


def create_stripe_connect_dashboard_link(
    user_id: str, user_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a link to the Stripe Express dashboard.

    Args:
        user_id: User identifier
        user_context: User profile data

    Returns:
        Dictionary with dashboard link action or error
    """
    try:
        stripe_status = user_context.get("stripe_connect", {})

        if not stripe_status.get("connected"):
            return {
                "error": "You need to set up Stripe Connect first before accessing the dashboard."
            }

        if not stripe_status.get("can_accept_payments"):
            return {
                "error": "Your Stripe Connect account isn't ready yet. Please complete the onboarding process first."
            }

        return {
            "action": "create_stripe_connect_dashboard_link",
            "user_id": user_id,
            "message": "I'll open your Stripe dashboard where you can manage your payments, view transactions, and update your account settings.",
        }

    except Exception as e:
        logger.error(f"Error creating dashboard link: {e}")
        return {"error": f"Error creating dashboard link: {str(e)}"}


def get_stripe_connect_earnings(
    user_id: str, user_context: Dict[str, Any], period: str = "month"
) -> Dict[str, Any]:
    """
    Get earnings information from Stripe Connect.

    Args:
        user_id: User identifier
        user_context: User profile data
        period: Time period for an earnings report

    Returns:
        Dictionary with earnings information or error
    """
    try:
        stripe_status = user_context.get("stripe_connect", {})

        if not stripe_status.get("connected"):
            return {
                "error": "You need to set up Stripe Connect first to view earnings."
            }

        # This would typically fetch real data from Stripe
        # For now, return a placeholder structure
        return {
            "action": "get_stripe_connect_earnings",
            "period": period,
            "user_id": user_id,
            "message": f"I'll get your Stripe Connect earnings for the {period}. This includes your payment totals and platform fees.",
        }

    except Exception as e:
        logger.error(f"Error getting Stripe Connect earnings: {e}")
        return {"error": f"Error getting Stripe Connect earnings: {str(e)}"}


def disconnect_stripe_connect(
    user_id: str, user_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Disconnect the user's Stripe Connect account.

    Args:
        user_id: User identifier
        user_context: User profile data

    Returns:
        Dictionary with disconnect action or error
    """
    try:
        stripe_status = user_context.get("stripe_connect", {})

        if not stripe_status.get("connected"):
            return {"error": "You don't have a Stripe Connect account to disconnect."}

        return {
            "action": "disconnect_stripe_connect",
            "user_id": user_id,
            "message": "I'll disconnect your Stripe Connect account. This will disable payment processing for your invoices.",
            "requires_confirmation": True,
            "warning": "After disconnecting, you won't be able to accept payments through Cashly until you reconnect.",
        }

    except Exception as e:
        logger.error(f"Error disconnecting Stripe Connect: {e}")
        return {"error": f"Error disconnecting Stripe Connect: {str(e)}"}

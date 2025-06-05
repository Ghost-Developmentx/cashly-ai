import pytest

from app.services.openai_assistants.assistant_manager.context_enhancer import ContextEnhancer


def test_enhance_query_no_user_context():
    """When no context is provided the query is unchanged."""
    query = "What is my balance?"
    result = ContextEnhancer.enhance_query(query)

    assert result.original_query == query
    assert result.enhanced_query == query
    assert result.context_parts == []
    assert result.additional_instructions == ""
    assert result.has_context is False


def test_enhance_query_with_user_context():
    """Context parts and instructions are added when context is provided."""
    user_context = {
        "accounts": [
            {"id": "1", "balance": 1000.0},
            {"id": "2", "balance": 500.0},
        ],
        "stripe_connect": {
            "connected": True,
            "can_accept_payments": False,
            "status": "pending",
        },
        "integrations": [
            {"provider": "Xero"},
            {"provider": "QuickBooks"},
        ],
        "transactions": [
            {"amount": 10},
            {"amount": 20},
        ],
    }

    query = "Generate financial report"
    result = ContextEnhancer.enhance_query(query, user_context)

    expected_context_parts = [
        "Connected accounts: 2 accounts with total balance of $1,500.00",
        "Stripe Connect: Connected but status is pending",
        "Active integrations: Xero, QuickBooks",
        "Available transaction data: 2 transactions",
    ]
    assert result.context_parts == expected_context_parts
    assert result.has_context is True
    assert result.enhanced_query.startswith("User context:")
    assert result.enhanced_query.endswith(f"User query: {query}")

    expected_instructions = (
        "Note: User has Stripe Connect but needs to complete setup. "
        "Invoices can be created but may need manual payment processing."
    )
    assert result.additional_instructions == expected_instructions


def test_generate_instructions_no_accounts():
    """Instructions mention missing accounts and Stripe setup."""
    context = {
        "accounts": [],
        "stripe_connect": {"connected": False},
        "transactions": [],
    }

    instructions = ContextEnhancer._generate_instructions(context)

    expected = (
        "Note: User has no connected bank accounts. "
        "Suggest connecting accounts for better insights. "
        "Note: User doesn't have Stripe Connect set up. "
        "For invoice queries, suggest setting up Stripe Connect. "
        "Note: No transaction history available. "
        "Financial insights will be limited."
    )
    assert instructions == expected


def test_generate_instructions_single_account_complete_setup():
    """Instructions for a single account with Stripe ready."""
    context = {
        "accounts": [{"id": "1"}],
        "stripe_connect": {"connected": True, "can_accept_payments": True},
        "transactions": [],
    }

    instructions = ContextEnhancer._generate_instructions(context)

    expected = (
        "Note: User has one connected account. "
        "Consider suggesting additional accounts for comprehensive tracking. "
        "Note: User has Stripe Connect set up and can accept payments. "
        "Invoices can be created and sent with payment links. "
        "Note: No transaction history available. "
        "Financial insights will be limited."
    )
    assert instructions == expected
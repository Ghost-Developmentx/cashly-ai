import pytest

from app.services.embeddings.context_builder import ConversationContextBuilder


def test_build_context_truncates_and_formats():
    builder = ConversationContextBuilder(max_messages=3, max_tokens=30)
    long_response = "x" * 220 + "END"
    history = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": long_response},
        {"role": "user", "content": "Thanks please show my balance."},
        {"role": "assistant", "content": "Sure, found 3 accounts."},
    ]

    context = builder.build_context(history, "What's my last transaction?")

    # Only last three messages should be included
    assert "First message" not in context
    # Assistant response should be summarized and contain ellipsis
    assert "..." in context
    # Ensure Current query is appended
    assert "Current query: What's my last transaction?" in context
    # Truncated to max token limit (approx chars)
    assert len(context) <= builder.max_tokens * 4


def test_extract_metadata_populates_fields():
    builder = ConversationContextBuilder()
    history = [
        {"role": "user", "content": "Can you show my account balance and latest transactions?"},
        {"role": "assistant", "content": "Found 5 transactions and your balance is $100."},
        {"role": "user", "content": "Please update the second transaction amount."},
        {"role": "assistant", "content": "Updated the transaction as requested."},
    ]

    metadata = builder.extract_metadata(history, "check_account", "chatbot")

    assert metadata["intent"] == "check_account"
    assert metadata["assistant_type"] == "chatbot"
    assert metadata["message_count"] == len(history)
    assert "timestamp" in metadata
    # Topics should include both accounts and transactions
    assert set(metadata.get("topics", [])) >= {"accounts", "transactions"}
    # Function calls should include retrieval and modification
    assert set(metadata.get("function_calls", [])) == {"data_retrieval", "data_modification"}
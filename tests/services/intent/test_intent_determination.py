import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.intent_determination.context_processor import ContextProcessor
from app.services.intent_determination.context_cache import AsyncContextCache
from app.services.intent_determination.intent_determiner import IntentDeterminer
from app.services.intent_determination.intent_resolver import AsyncIntentResolver
from app.services.search.async_vector_search import SearchResult


CONVERSATION = [
    {"role": "user", "content": "Can you show my recent transactions?"},
    {"role": "assistant", "content": "Sure, I found 5 transactions."},
    {"role": "user", "content": "Create an invoice for client A."},
    {"role": "assistant", "content": "Invoice created successfully."},
]

USER_CONTEXT = {
    "accounts": [{"balance": 1000}, {"balance": 500}],
    "stripe_connect": {"connected": True, "can_accept_payments": True},
    "transactions": [{"id": 1}, {"id": 2}],
}


def test_extract_key_information():
    processor = ContextProcessor()
    info = processor.extract_key_information(CONVERSATION, USER_CONTEXT)

    assert info["user_queries"] == [
        "Can you show my recent transactions?",
        "Create an invoice for client A.",
    ]
    assert set(info["topics"]) == {"transactions", "invoices"}
    assert info["actions_taken"] == [
        {"type": "retrieve", "inferred": True},
        {"type": "create", "inferred": True},
    ]
    assert info["user_context_summary"]["connected_accounts"] == 2
    assert info["user_context_summary"]["total_balance"] == 1500
    assert info["message_count"] == 4


def test_prepare_embedding_text_components():
    processor = ContextProcessor()
    text = processor.prepare_embedding_text(
        CONVERSATION, USER_CONTEXT, current_query="What's my balance?"
    )

    assert "User Context:" in text
    assert "Conversation:" in text
    assert "Current Query: What's my balance?" in text


def test_prepare_embedding_text_truncates():
    processor = ContextProcessor()
    text = processor.prepare_embedding_text(
        CONVERSATION,
        USER_CONTEXT,
        current_query="test",
        max_length=30,
    )
    assert len(text) <= 30
    assert text.endswith("...")


@pytest.mark.asyncio
async def test_context_cache_expiry():
    cache = AsyncContextCache(ttl_minutes=0.001, max_size=10)
    await cache.set("c1", {"a": 1})
    await asyncio.sleep(0.1)
    assert await cache.get("c1") is None


@pytest.mark.asyncio
async def test_context_cache_eviction():
    cache = AsyncContextCache(ttl_minutes=1, max_size=2)
    await cache.set("c1", {"v": 1})
    await cache.set("c2", {"v": 2})
    await cache.set("c3", {"v": 3})

    assert await cache.get("c1") is None
    assert await cache.get("c2") == {"v": 2}
    assert await cache.get("c3") == {"v": 3}


def test_intent_determiner_basic():
    determiner = IntentDeterminer()
    results = [
        SearchResult(
            conversation_id="1",
            intent="transactions",
            assistant_type="transaction_assistant",
            similarity_score=0.8,
            success_indicator=True,
            metadata={},
        ),
        SearchResult(
            conversation_id="2",
            intent="transactions",
            assistant_type="transaction_assistant",
            similarity_score=0.7,
            success_indicator=True,
            metadata={},
        ),
        SearchResult(
            conversation_id="3",
            intent="invoices",
            assistant_type="invoice_assistant",
            similarity_score=0.6,
            success_indicator=True,
            metadata={},
        ),
    ]

    intent, confidence, analysis = determiner.determine_intent(results)

    assert intent == "transactions"
    assert 0 < confidence <= 0.95
    assert analysis["search_results_count"] == 3


@pytest.mark.asyncio
async def test_async_intent_resolver_happy_path():
    resolver = AsyncIntentResolver()

    resolver.context_aggregator = AsyncMock()
    resolver.context_aggregator.process_conversation_async = AsyncMock(
        return_value={"embedding_text": "text", "key_information": {}}
    )

    resolver.embedding_client = AsyncMock()
    resolver.embedding_client.create_embedding = AsyncMock(return_value=[0.1, 0.2])

    search_result = SearchResult(
        conversation_id="1",
        intent="transactions",
        assistant_type="transaction_assistant",
        similarity_score=0.9,
        success_indicator=True,
        metadata={},
    )
    resolver.search_service = AsyncMock()
    resolver.search_service.search_similar = AsyncMock(return_value=[search_result])

    resolver.intent_determiner = MagicMock()
    resolver.intent_determiner.determine_intent.return_value = (
        "transactions",
        0.9,
        {
            "method": "test",
            "evidence_count": 1,
            "success_rate": 1.0,
            "alternatives": [],
            "search_results_count": 1,
            "avg_similarity": 0.9,
        },
    )

    resolver.routing_intelligence = MagicMock()
    resolver.routing_intelligence.recommend_assistant.return_value = {
        "assistant": "transaction_assistant",
        "confidence": 0.8,
    }

    resolution = await resolver.resolve_intent(
        query="show me transactions",
        conversation_history=[],
        user_id="u1",
        conversation_id="c1",
        user_context={},
    )

    assert resolution["intent"] == "transactions"
    assert resolution["recommended_assistant"] == "transaction_assistant"
    assert resolution["confidence"] == 0.9
    assert resolution["analysis"]["routing"]["assistant"] == "transaction_assistant"
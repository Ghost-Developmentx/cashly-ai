import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.services.openai_assistants.assistant_manager.thread_manager import ThreadManager


class MockAsyncOpenAI:
    """Simple mock of AsyncOpenAI with minimal beta thread features."""

    def __init__(self, *args, **kwargs):
        self.thread_create_mock = AsyncMock(return_value=SimpleNamespace(id="mock_thread_1"))
        self.message_create_mock = AsyncMock(return_value=SimpleNamespace(id="msg1"))
        self.message_list_mock = AsyncMock(return_value=SimpleNamespace(data=[]))
        self.beta = SimpleNamespace(
            threads=SimpleNamespace(
                create=self.thread_create_mock,
                messages=SimpleNamespace(
                    create=self.message_create_mock, list=self.message_list_mock
                ),
            )
        )


@pytest.mark.asyncio
async def test_thread_lifecycle(monkeypatch):
    """ThreadManager should create, reuse and clear threads properly."""

    # Patch AsyncOpenAI used inside ThreadManager
    monkeypatch.setattr("openai.AsyncOpenAI", MockAsyncOpenAI)

    manager = ThreadManager()
    user_id = "user1"

    # First call creates a thread
    thread_id1 = await manager.get_or_create_thread(user_id)
    assert thread_id1 == "mock_thread_1"
    assert manager._active_threads[user_id] == thread_id1
    assert manager._thread_metadata[thread_id1]["message_count"] == 0

    # Second call should reuse existing thread (no new create)
    thread_id2 = await manager.get_or_create_thread(user_id)
    assert thread_id2 == thread_id1
    assert manager.client.thread_create_mock.await_count == 1

    # Add two messages and verify metadata increments
    await manager.add_message(thread_id1, "hi")
    await manager.add_message(thread_id1, "there")
    assert manager._thread_metadata[thread_id1]["message_count"] == 2

    # Clear thread should remove stored data
    assert manager.clear_thread(user_id) is True
    assert user_id not in manager._active_threads
    assert thread_id1 not in manager._thread_metadata

    # After clearing, creating again should call create once more
    manager.client.thread_create_mock.return_value = SimpleNamespace(id="mock_thread_2")
    new_thread_id = await manager.get_or_create_thread(user_id)
    assert new_thread_id == "mock_thread_2"
    assert manager.client.thread_create_mock.await_count == 2
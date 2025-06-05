import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from app.services.openai_assistants.assistant_manager.response_handler import ResponseHandler
from app.services.openai_assistants.assistant_manager.types import AssistantType, AssistantResponse


def test_extract_message_content():
    """_extract_message_content should join text blocks in order."""
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text=SimpleNamespace(value="first")),
            SimpleNamespace(type="image_url", text=SimpleNamespace(value="ignored")),
            SimpleNamespace(type="text", text=SimpleNamespace(value="second")),
        ]
    )

    result = ResponseHandler._extract_message_content(message)
    assert result == "first\nsecond"


def test_create_error_response():
    """_create_error_response should populate AssistantResponse properly."""
    metadata = {"foo": "bar"}
    function_calls = [{"name": "call"}]

    resp = ResponseHandler._create_error_response(
        AssistantType.GENERAL,
        function_calls,
        "oops",
        metadata,
    )

    assert resp.success is False
    assert resp.error == "oops"
    assert resp.function_calls == function_calls
    assert resp.assistant_type == AssistantType.GENERAL
    assert resp.metadata["foo"] == "bar"
    assert "error_at" in resp.metadata


@pytest.mark.asyncio
async def test_wait_for_run_completion_completed(monkeypatch):
    """wait_for_run_completion should return response from _handle_completed_run when run is completed."""
    handler = ResponseHandler.__new__(ResponseHandler)
    handler.config = SimpleNamespace(timeout=1)
    handler.client = MagicMock()
    handler.client.beta.threads.runs.retrieve = AsyncMock(
        return_value=SimpleNamespace(status="completed")
    )

    expected = AssistantResponse(
        content="done",
        assistant_type=AssistantType.GENERAL,
        function_calls=[],
        metadata={"run_id": "run123", "thread_id": "thread123"},
        success=True,
    )

    async def mock_completed(thread_id, run_id, assistant_type, function_calls):
        assert thread_id == "thread123"
        assert run_id == "run123"
        assert assistant_type == AssistantType.GENERAL
        assert function_calls == []
        return expected

    handler._handle_completed_run = mock_completed

    resp = await handler.wait_for_run_completion(
        thread_id="thread123",
        run_id="run123",
        assistant_type=AssistantType.GENERAL,
    )

    handler.client.beta.threads.runs.retrieve.assert_called_once_with(
        thread_id="thread123", run_id="run123"
    )
    assert resp == expected
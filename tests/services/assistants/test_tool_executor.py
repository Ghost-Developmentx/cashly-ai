import json
import pytest

from app.services.openai_assistants.assistant_manager.tool_executor import ToolExecutor


class FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id_: str, function_name: str, arguments: str):
        self.id = id_
        self.function = FakeFunction(function_name, arguments)


@pytest.fixture
def executor(monkeypatch):
    """Create ToolExecutor without real OpenAI client."""
    monkeypatch.setattr(
        "app.services.openai_assistants.assistant_manager.base.AsyncOpenAI",
        lambda *args, **kwargs: None,
    )
    return ToolExecutor()


@pytest.mark.asyncio
async def test_execute_tool_calls_success(executor):
    async def fake_executor(name, args, **context):
        return {"ran": name, **args}

    executor.set_tool_executor(fake_executor)

    tool_call = FakeToolCall("123", "do_something", "{\"value\": 1}")

    outputs, calls = await executor.execute_tool_calls([tool_call], "user1")

    assert isinstance(outputs, list) and isinstance(calls, list)
    assert len(outputs) == 1 and len(calls) == 1

    expected_result = {"ran": "do_something", "value": 1}
    assert outputs[0]["tool_call_id"] == "123"
    assert json.loads(outputs[0]["output"]) == expected_result

    assert calls[0]["function"] == "do_something"
    assert calls[0]["arguments"] == {"value": 1}
    assert calls[0]["result"] == expected_result


@pytest.mark.asyncio
async def test_execute_tool_calls_error(executor):
    async def failing_executor(name, args, **context):
        raise RuntimeError("boom")

    executor.set_tool_executor(failing_executor)

    tool_call = FakeToolCall("err", "fail", "{}")

    outputs, calls = await executor.execute_tool_calls([tool_call], "user1")

    expected_output, expected_call = ToolExecutor._create_error_result(tool_call, "boom")

    assert outputs == [expected_output]
    assert calls == [expected_call]
"""Tests for the Meeseeks API backend."""
# mypy: ignore-errors
import json
import os
import sys
import types

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

langchain_community = types.ModuleType("langchain_community")
document_loaders = types.ModuleType("langchain_community.document_loaders")
document_loaders.JSONLoader = object
langchain_community.document_loaders = document_loaders
sys.modules["langchain_community"] = langchain_community
sys.modules["langchain_community.document_loaders"] = document_loaders

langchain_openai = types.ModuleType("langchain_openai")
langchain_openai.ChatOpenAI = object
sys.modules["langchain_openai"] = langchain_openai

langfuse = types.ModuleType("langfuse")
langfuse_callback = types.ModuleType("langfuse.callback")
langfuse_callback.CallbackHandler = object
langfuse.callback = langfuse_callback
sys.modules["langfuse"] = langfuse
sys.modules["langfuse.callback"] = langfuse_callback

langchain_core = types.ModuleType("langchain_core")
langchain_core_messages = types.ModuleType("langchain_core.messages")
langchain_core_messages_ai = types.ModuleType("langchain_core.messages.ai")
langchain_core_api = types.ModuleType("langchain_core._api.beta_decorator")
langchain_core_messages.SystemMessage = object
langchain_core_messages.HumanMessage = object
langchain_core_messages_ai.AIMessage = object
langchain_core_api.LangChainBetaWarning = Warning
langchain_core.messages = langchain_core_messages
langchain_core._api = types.ModuleType("langchain_core._api")
langchain_core._api.beta_decorator = langchain_core_api
sys.modules["langchain_core"] = langchain_core
sys.modules["langchain_core.messages"] = langchain_core_messages
sys.modules["langchain_core.messages.ai"] = langchain_core_messages_ai
sys.modules["langchain_core._api.beta_decorator"] = langchain_core_api

langchain = types.ModuleType("langchain")
langchain_prompts = types.ModuleType("langchain.prompts")
langchain_output_parsers = types.ModuleType("langchain.output_parsers")
langchain_prompts.ChatPromptTemplate = object
langchain_prompts.HumanMessagePromptTemplate = object
langchain_output_parsers.PydanticOutputParser = object
langchain.prompts = langchain_prompts
langchain.output_parsers = langchain_output_parsers
sys.modules["langchain"] = langchain
sys.modules["langchain.prompts"] = langchain_prompts
sys.modules["langchain.output_parsers"] = langchain_output_parsers

langchain_core_pydantic = types.ModuleType("langchain_core.pydantic_v1")

class BaseModel:
    """Minimal stand-in for Pydantic BaseModel."""
    def __init__(self, **kwargs):
        """Store provided attributes on the instance."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def dict(self, *args, **kwargs):
        """Return a shallow dict representation of the instance."""
        return dict(self.__dict__)

def Field(*args, **kwargs):  # noqa: N802 - mimic pydantic API
    """Stub Field helper for pydantic-compatible imports."""
    return None

def validator(*args, **kwargs):  # noqa: N802 - mimic pydantic API
    """Stub validator decorator for pydantic-compatible imports."""
    def wrapper(func):
        return func

    return wrapper

langchain_core_pydantic.BaseModel = BaseModel
langchain_core_pydantic.Field = Field
langchain_core_pydantic.validator = validator
langchain_core.pydantic_v1 = langchain_core_pydantic
sys.modules["langchain_core.pydantic_v1"] = langchain_core_pydantic

dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: None
sys.modules["dotenv"] = dotenv

coloredlogs = types.ModuleType("coloredlogs")
coloredlogs.install = lambda *args, **kwargs: None
sys.modules["coloredlogs"] = coloredlogs

tiktoken = types.ModuleType("tiktoken")
tiktoken.get_encoding = lambda *args, **kwargs: types.SimpleNamespace(
    encode=lambda value: list(value)
)
sys.modules["tiktoken"] = tiktoken


import backend  # noqa: E402


class DummyQueue:
    """Minimal task queue stub for API responses."""
    def __init__(self, result: str) -> None:
        """Initialize the dummy queue with a single action result."""
        self.task_result = result
        self.action_steps = [
            {
                "action_consumer": "talk_to_user_tool",
                "action_type": "set",
                "action_argument": "say",
                "result": result,
            }
        ]

    def dict(self):
        """Return a serialized representation of the queue."""
        return {
            "task_result": self.task_result,
            "action_steps": list(self.action_steps),
        }


def _make_task_queue(result: str) -> DummyQueue:
    return DummyQueue(result)


def test_query_requires_api_key(monkeypatch):
    """Require authentication headers for query requests."""
    client = backend.app.test_client()
    response = client.post("/api/query", json={"query": "hello"})
    assert response.status_code == 401


def test_query_invalid_input(monkeypatch):
    """Reject empty payloads without a query value."""
    client = backend.app.test_client()
    response = client.post(
        "/api/query",
        headers={"X-API-KEY": backend.MASTER_API_TOKEN},
        data=json.dumps({}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_query_success(monkeypatch):
    """Return a task result payload when authorized."""
    client = backend.app.test_client()

    def fake_orchestrate(*args, **kwargs):
        return _make_task_queue("ok")

    monkeypatch.setattr(backend, "orchestrate_session", fake_orchestrate)
    response = client.post(
        "/api/query",
        headers={"X-API-KEY": backend.MASTER_API_TOKEN},
        json={"query": "hello"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["task_result"] == "ok"
    assert payload["session_id"]
    assert payload["action_steps"]

"""Tests for the Streamlit chat UI helpers."""
# ruff: noqa: I001
# mypy: ignore-errors
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
langchain_memory = types.ModuleType("langchain.memory")
langchain_prompts.ChatPromptTemplate = object
langchain_prompts.HumanMessagePromptTemplate = object
langchain_output_parsers.PydanticOutputParser = object
langchain_memory.ConversationBufferWindowMemory = object
langchain.prompts = langchain_prompts
langchain.output_parsers = langchain_output_parsers
langchain.memory = langchain_memory
sys.modules["langchain"] = langchain
sys.modules["langchain.prompts"] = langchain_prompts
sys.modules["langchain.output_parsers"] = langchain_output_parsers
sys.modules["langchain.memory"] = langchain_memory

langchain_core_pydantic = types.ModuleType("langchain_core.pydantic_v1")

class BaseModel:
    """Minimal stand-in for Pydantic BaseModel."""
    def __init__(self, **kwargs):
        """Store provided attributes on the instance."""
        for key, value in kwargs.items():
            setattr(self, key, value)

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

jinja2 = types.ModuleType("jinja2")
jinja2.Environment = object
jinja2.FileSystemLoader = object
sys.modules["jinja2"] = jinja2

streamlit = types.ModuleType("streamlit")
streamlit.session_state = types.SimpleNamespace()
streamlit.set_page_config = lambda *args, **kwargs: None
streamlit.image = lambda *args, **kwargs: None
streamlit.markdown = lambda *args, **kwargs: None
streamlit.chat_message = types.SimpleNamespace
streamlit.chat_input = lambda *args, **kwargs: None
streamlit.spinner = types.SimpleNamespace
streamlit.expander = types.SimpleNamespace
streamlit.caption = lambda *args, **kwargs: None
sys.modules["streamlit"] = streamlit

import chat_master  # noqa: E402
from core.classes import ActionStep, TaskQueue  # noqa: E402
from core.session_store import SessionStore  # noqa: E402


def _make_task_queue(action_steps):
    task_queue = TaskQueue(action_steps=action_steps)
    task_queue.human_message = "hello"
    return task_queue


def test_generate_action_plan_helper(monkeypatch):
    """Return a formatted action plan with the generated queue."""
    steps = [
        ActionStep(
            action_consumer="talk_to_user_tool",
            action_type="set",
            action_argument="hello",
        )
    ]
    task_queue = _make_task_queue(steps)

    def fake_generate(*args, **kwargs):
        return task_queue

    monkeypatch.setattr(chat_master, "generate_action_plan", fake_generate)
    plan, returned = chat_master.generate_action_plan_helper("hello")
    assert returned == task_queue
    assert plan == [
        "Using `talk_to_user_tool` with `set` to `hello`",
    ]


def test_run_action_plan_helper(monkeypatch, tmp_path):
    """Combine responses from a task queue execution."""
    session_store = SessionStore(root_dir=str(tmp_path))
    session_id = session_store.create_session()
    fake_state = types.SimpleNamespace(
        session_store=session_store,
        session_id=session_id,
    )
    monkeypatch.setattr(chat_master, "st", types.SimpleNamespace(session_state=fake_state))

    steps = [
        ActionStep(
            action_consumer="talk_to_user_tool",
            action_type="set",
            action_argument="hello",
        )
    ]
    task_queue = _make_task_queue(steps)

    def fake_orchestrate(*args, **kwargs):
        MockSpeaker = type("Result", (), {"content": "ok"})
        task_queue.action_steps[0].result = MockSpeaker()
        task_queue.task_result = "ok"
        return task_queue

    monkeypatch.setattr(chat_master, "orchestrate_session", fake_orchestrate)
    response = chat_master.run_action_plan_helper(task_queue)
    assert response == "ok"


def test_main_flow(monkeypatch, tmp_path):
    """Exercise the main chat flow with stubbed Streamlit runtime."""
    class DummyContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeSessionState(dict):
        def __getattr__(self, item):
            return self[item]

        def __setattr__(self, key, value):
            self[key] = value

    class FakeStreamlit:
        def __init__(self):
            self.session_state = FakeSessionState()

        def set_page_config(self, *args, **kwargs):
            return None

        def image(self, *args, **kwargs):
            return None

        def markdown(self, *args, **kwargs):
            return None

        def chat_message(self, *args, **kwargs):
            return DummyContext()

        def chat_input(self, *args, **kwargs):
            return "hello"

        def spinner(self, *args, **kwargs):
            return DummyContext()

        def expander(self, *args, **kwargs):
            return DummyContext()

        def caption(self, *args, **kwargs):
            return None

    class DummyMemory:
        def __init__(self, k=5):
            self.saved = []

        def save_context(self, inputs, outputs):
            self.saved.append((inputs, outputs))

    fake_st = FakeStreamlit()
    monkeypatch.setattr(chat_master, "st", fake_st)
    monkeypatch.setattr(chat_master, "ConversationBufferWindowMemory", DummyMemory)
    monkeypatch.setattr(chat_master.time, "sleep", lambda *_: None)

    def fake_generate(user_input):
        steps = [
            ActionStep(
                action_consumer="talk_to_user_tool",
                action_type="set",
                action_argument="hello",
            )
        ]
        return ["Using `talk_to_user_tool` with `set` to `hello`"], TaskQueue(
            action_steps=steps
        )

    monkeypatch.setattr(chat_master, "generate_action_plan_helper", fake_generate)
    monkeypatch.setattr(chat_master, "run_action_plan_helper", lambda *_: "ok")
    monkeypatch.setattr(chat_master, "SessionStore", lambda: SessionStore(root_dir=str(tmp_path)))

    chat_master.main()

    assert "messages" in fake_st.session_state
    assert any(msg["role"] == "assistant" for msg in fake_st.session_state["messages"])

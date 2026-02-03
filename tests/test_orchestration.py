import sys
import types

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
langchain_core_pydantic = types.ModuleType("langchain_core.pydantic_v1")

dummy_pydantic = types.ModuleType("pydantic")

class BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def Field(*args, **kwargs):  # noqa: N802 - mimic pydantic API
    return None

def validator(*args, **kwargs):  # noqa: N802 - mimic pydantic API
    def wrapper(func):
        return func

    return wrapper

dummy_pydantic.BaseModel = BaseModel
dummy_pydantic.Field = Field
dummy_pydantic.validator = validator
sys.modules["pydantic"] = dummy_pydantic

langchain_core_pydantic.BaseModel = BaseModel
langchain_core_pydantic.Field = Field
langchain_core_pydantic.validator = validator

langchain_core_messages.SystemMessage = object
langchain_core_messages.HumanMessage = object
langchain_core_messages_ai.AIMessage = object
langchain_core_api.LangChainBetaWarning = Warning

langchain_core.messages = langchain_core_messages
langchain_core.pydantic_v1 = langchain_core_pydantic
langchain_core._api = types.ModuleType("langchain_core._api")
langchain_core._api.beta_decorator = langchain_core_api

sys.modules["langchain_core"] = langchain_core
sys.modules["langchain_core.messages"] = langchain_core_messages
sys.modules["langchain_core.messages.ai"] = langchain_core_messages_ai
sys.modules["langchain_core._api.beta_decorator"] = langchain_core_api
sys.modules["langchain_core.pydantic_v1"] = langchain_core_pydantic

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

dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *args, **kwargs: None
sys.modules["dotenv"] = dotenv

jinja2 = types.ModuleType("jinja2")
jinja2.Environment = object
jinja2.FileSystemLoader = object
sys.modules["jinja2"] = jinja2

coloredlogs = types.ModuleType("coloredlogs")
coloredlogs.install = lambda *args, **kwargs: None
sys.modules["coloredlogs"] = coloredlogs

tiktoken = types.ModuleType("tiktoken")

class DummyEncoding:
    def encode(self, string):
        return list(string)

tiktoken.get_encoding = lambda *args, **kwargs: DummyEncoding()
sys.modules["tiktoken"] = tiktoken

requests = types.ModuleType("requests")
requests.exceptions = types.SimpleNamespace(RequestException=Exception)
requests.get = lambda *args, **kwargs: types.SimpleNamespace(
    raise_for_status=lambda: None, json=lambda: []
)
requests.post = lambda *args, **kwargs: types.SimpleNamespace(
    raise_for_status=lambda: None, json=lambda: [], text=""
)
sys.modules["requests"] = requests

from core import task_master  # noqa: E402
from core.classes import ActionStep, TaskQueue  # noqa: E402
from core.common import get_mock_speaker  # noqa: E402
from core.hooks import HookManager  # noqa: E402
from core.permissions import (  # noqa: E402
    PermissionDecision,
    PermissionPolicy,
    PermissionRule,
)
from core.session_store import SessionStore  # noqa: E402
from core.tool_registry import ToolRegistry, ToolSpec  # noqa: E402


class Counter:
    def __init__(self):
        self.count = 0

    def bump(self):
        self.count += 1


def make_task_queue(message: str) -> TaskQueue:
    step = ActionStep(
        action_consumer="talk_to_user_tool",
        action_type="set",
        action_argument=message,
    )
    return TaskQueue(action_steps=[step])


def test_orchestrate_session_completes(monkeypatch, tmp_path):
    generate_calls = Counter()
    run_calls = Counter()
    session_store = SessionStore(root_dir=str(tmp_path))
    session_id = session_store.create_session()

    def fake_generate(*args, **kwargs):
        generate_calls.bump()
        return make_task_queue("say hi")

    def fake_run(task_queue, **kwargs):
        run_calls.bump()
        MockSpeaker = get_mock_speaker()
        task_queue.action_steps[0].result = MockSpeaker(content="done")
        task_queue.task_result = "done"
        return task_queue

    monkeypatch.setattr(task_master, "generate_action_plan", fake_generate)
    monkeypatch.setattr(task_master, "run_action_plan", fake_run)

    task_queue = task_master.orchestrate_session(
        "hello",
        max_iters=3,
        session_id=session_id,
        session_store=session_store,
    )

    assert task_queue.task_result == "done"
    assert generate_calls.count == 1
    assert run_calls.count == 1


def test_orchestrate_session_replans_on_failure(monkeypatch, tmp_path):
    generate_calls = Counter()
    run_calls = Counter()
    session_store = SessionStore(root_dir=str(tmp_path))
    session_id = session_store.create_session()

    def fake_generate(*args, **kwargs):
        generate_calls.bump()
        return make_task_queue("say hi")

    def fake_run(task_queue, **kwargs):
        run_calls.bump()
        if run_calls.count == 1:
            task_queue.action_steps[0].result = None
            task_queue.task_result = "failed"
        else:
            MockSpeaker = get_mock_speaker()
            task_queue.action_steps[0].result = MockSpeaker(content="ok")
            task_queue.task_result = "ok"
        return task_queue

    monkeypatch.setattr(task_master, "generate_action_plan", fake_generate)
    monkeypatch.setattr(task_master, "run_action_plan", fake_run)

    task_queue = task_master.orchestrate_session(
        "hello",
        max_iters=2,
        session_id=session_id,
        session_store=session_store,
    )

    assert task_queue.task_result == "ok"
    assert generate_calls.count == 2
    assert run_calls.count == 2


def test_orchestrate_session_passes_summary(monkeypatch, tmp_path):
    captured = {}
    session_store = SessionStore(root_dir=str(tmp_path))
    session_id = session_store.create_session()
    session_store.save_summary(session_id, "previous summary")

    def fake_generate(*args, **kwargs):
        captured["summary"] = kwargs.get("session_summary")
        return make_task_queue("say hi")

    def fake_run(task_queue, **kwargs):
        MockSpeaker = get_mock_speaker()
        task_queue.action_steps[0].result = MockSpeaker(content="ok")
        task_queue.task_result = "ok"
        return task_queue

    monkeypatch.setattr(task_master, "generate_action_plan", fake_generate)
    monkeypatch.setattr(task_master, "run_action_plan", fake_run)

    task_master.orchestrate_session(
        "hello",
        max_iters=1,
        session_id=session_id,
        session_store=session_store,
    )

    assert captured["summary"] == "previous summary"


def test_orchestrate_session_max_iters(monkeypatch, tmp_path):
    generate_calls = Counter()
    run_calls = Counter()
    session_store = SessionStore(root_dir=str(tmp_path))
    session_id = session_store.create_session()

    def fake_generate(*args, **kwargs):
        generate_calls.bump()
        return make_task_queue("say hi")

    def fake_run(task_queue, **kwargs):
        run_calls.bump()
        task_queue.action_steps[0].result = None
        task_queue.task_result = "failed"
        return task_queue

    monkeypatch.setattr(task_master, "generate_action_plan", fake_generate)
    monkeypatch.setattr(task_master, "run_action_plan", fake_run)

    task_queue, state = task_master.orchestrate_session(
        "hello",
        max_iters=1,
        return_state=True,
        session_id=session_id,
        session_store=session_store,
    )

    assert task_queue.task_result == "failed"
    assert state.done is False
    assert state.done_reason == "max_iterations_reached"
    assert generate_calls.count == 1
    assert run_calls.count == 1


def test_orchestrate_session_compact(tmp_path):
    session_store = SessionStore(root_dir=str(tmp_path))
    session_id = session_store.create_session()
    session_store.append_event(
        session_id,
        {"type": "user", "payload": {"text": "hello"}},
    )

    task_queue, state = task_master.orchestrate_session(
        "/compact",
        return_state=True,
        session_id=session_id,
        session_store=session_store,
    )

    assert state.done is True
    assert state.done_reason == "compacted"
    assert task_queue.task_result is not None


class DummyTool:
    def __init__(self):
        self.called_with = None

    def run(self, action_step):
        self.called_with = action_step.action_argument
        MockSpeaker = get_mock_speaker()
        return MockSpeaker(content=f"ok:{action_step.action_argument}")


def test_run_action_plan_permission_denied():
    registry = ToolRegistry()
    dummy_tool = DummyTool()
    registry.register(
        ToolSpec(
            tool_id="dummy_tool",
            name="Dummy",
            description="Dummy tool",
            factory=lambda: dummy_tool,
        )
    )
    policy = PermissionPolicy(
        rules=[
            PermissionRule(
                tool_id="dummy_tool",
                action_type="set",
                decision=PermissionDecision.DENY,
            )
        ],
        default_by_action={"get": PermissionDecision.ALLOW, "set": PermissionDecision.ASK},
        default_decision=PermissionDecision.ASK,
    )
    task_queue = TaskQueue(
        action_steps=[
            ActionStep(
                action_consumer="dummy_tool",
                action_type="set",
                action_argument="payload",
            )
        ]
    )
    task_queue = task_master.run_action_plan(
        task_queue,
        tool_registry=registry,
        permission_policy=policy,
        approval_callback=lambda _: False,
    )
    assert dummy_tool.called_with is None
    assert "Permission denied" in (task_queue.task_result or "")


def test_run_action_plan_hooks_modify_input():
    registry = ToolRegistry()
    dummy_tool = DummyTool()
    registry.register(
        ToolSpec(
            tool_id="dummy_tool",
            name="Dummy",
            description="Dummy tool",
            factory=lambda: dummy_tool,
        )
    )
    policy = PermissionPolicy(
        rules=[],
        default_by_action={"get": PermissionDecision.ALLOW, "set": PermissionDecision.ALLOW},
        default_decision=PermissionDecision.ALLOW,
    )

    def pre_hook(step):
        step.action_argument = "updated"
        return step

    def post_hook(step, result):
        MockSpeaker = get_mock_speaker()
        return MockSpeaker(content=f"post:{step.action_argument}")

    hooks = HookManager(pre_tool_use=[pre_hook], post_tool_use=[post_hook])
    task_queue = TaskQueue(
        action_steps=[
            ActionStep(
                action_consumer="dummy_tool",
                action_type="get",
                action_argument="original",
            )
        ]
    )
    task_queue = task_master.run_action_plan(
        task_queue,
        tool_registry=registry,
        permission_policy=policy,
        hook_manager=hooks,
    )
    assert dummy_tool.called_with == "updated"
    assert task_queue.task_result == "post:updated"


def test_orchestrate_session_auto_compact(monkeypatch, tmp_path):
    session_store = SessionStore(root_dir=str(tmp_path))
    session_id = session_store.create_session()
    monkeypatch.setenv("MESEEKS_AUTO_COMPACT_THRESHOLD", "0")

    def fake_generate(*args, **kwargs):
        return make_task_queue("say hi")

    def fake_run(task_queue, **kwargs):
        MockSpeaker = get_mock_speaker()
        task_queue.action_steps[0].result = MockSpeaker(content="done")
        task_queue.task_result = "done"
        return task_queue

    monkeypatch.setattr(task_master, "generate_action_plan", fake_generate)
    monkeypatch.setattr(task_master, "run_action_plan", fake_run)

    task_master.orchestrate_session(
        "hello",
        max_iters=1,
        session_id=session_id,
        session_store=session_store,
    )

    assert session_store.load_summary(session_id) is not None

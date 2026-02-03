"""Tests for orchestration workflows."""
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
    """Simple counter helper for call tracking."""
    def __init__(self):
        """Initialize the counter."""
        self.count = 0

    def bump(self):
        """Increment the counter by one."""
        self.count += 1


def make_task_queue(message: str) -> TaskQueue:
    """Build a minimal task queue with a single action step."""
    step = ActionStep(
        action_consumer="talk_to_user_tool",
        action_type="set",
        action_argument=message,
    )
    return TaskQueue(action_steps=[step])


def test_orchestrate_session_completes(monkeypatch, tmp_path):
    """Return a completed task queue when execution succeeds."""
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
    """Replan when an action plan fails once."""
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
    """Pass stored summaries into action plan generation."""
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
    """Stop after max iteration count is reached."""
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
    """Compact session transcripts when limits are exceeded."""
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
    """Stub tool used for permission tests."""
    def __init__(self):
        """Initialize the dummy tool."""
        self.called_with = None

    def run(self, action_step):
        """Return a mocked tool response."""
        self.called_with = action_step.action_argument
        MockSpeaker = get_mock_speaker()
        return MockSpeaker(content=f"ok:{action_step.action_argument}")


def test_run_action_plan_permission_denied():
    """Return permission denied when policy disallows execution."""
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
    """Allow hooks to modify action arguments."""
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
    """Auto-compact sessions based on token budget."""
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

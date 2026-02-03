#!/usr/bin/env python3
"""Terminal CLI for Meeseeks."""

import argparse
import os
import sys
from collections.abc import Callable, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Add repo root to path for core imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cli_commands import get_registry  # noqa: E402
from cli_context import CliState, CommandContext  # noqa: E402

from core.classes import ActionStep, TaskQueue  # noqa: E402
from core.permissions import PermissionDecision  # noqa: E402
from core.session_store import SessionStore  # noqa: E402
from core.task_master import generate_action_plan, orchestrate_session  # noqa: E402
from core.tool_registry import ToolRegistry, load_registry  # noqa: E402


def _resolve_session_id(
    store: SessionStore,
    session_id: str | None,
    session_tag: str | None,
    fork_from: str | None,
) -> str:
    if fork_from:
        source_session_id = store.resolve_tag(fork_from) or fork_from
        session_id = store.fork_session(source_session_id)
    if session_tag and not session_id:
        resolved = store.resolve_tag(session_tag)
        session_id = resolved if resolved else None
    if not session_id:
        session_id = store.create_session()
    if session_tag:
        store.tag_session(session_id, session_tag)
    return session_id


def _format_steps(steps: Iterable[ActionStep]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for step in steps:
        rows.append((step.action_consumer, step.action_type, step.action_argument))
    return rows


def _ensure_history_path(path: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _parse_command(text: str) -> tuple[str, list[str]]:
    parts = text.strip().split()
    command = parts[0]
    return command, parts[1:]

def run_cli(args: argparse.Namespace) -> int:
    console = Console(color_system=None if args.no_color else "auto")
    store = SessionStore(root_dir=args.session_dir)
    session_id = _resolve_session_id(store, args.session, args.tag, args.fork)
    state = CliState(session_id=session_id, show_plan=args.show_plan, model_name=args.model)
    tool_registry = load_registry()
    registry = get_registry()

    console.print(Panel("Meeseeks CLI ready", title="Meeseeks"))
    console.print(f"Session: {state.session_id}")
    console.print("Type /help for commands.\n")

    if args.query:
        return _run_single_query(console, store, state, tool_registry, args.query, args)

    history_path = _ensure_history_path(args.history_file)
    session: PromptSession[str] = PromptSession(history=FileHistory(history_path))

    while True:
        try:
            user_input = session.prompt("meeseeks> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return 0

        if not user_input:
            continue
        if user_input.startswith("/"):
            command, cmd_args = _parse_command(user_input)
            context = CommandContext(
                console=console,
                store=store,
                state=state,
                tool_registry=tool_registry,
                prompt_func=session.prompt,
            )
            if not registry.execute(command, context, cmd_args):
                return 0
            continue

        _run_query(console, store, state, tool_registry, user_input, args, session.prompt)


def _run_single_query(
    console: Console,
    store: SessionStore,
    state: CliState,
    tool_registry: ToolRegistry,
    query: str,
    args: argparse.Namespace,
) -> int:
    _run_query(console, store, state, tool_registry, query, args, None)
    return 0


def _run_query(
    console: Console,
    store: SessionStore,
    state: CliState,
    tool_registry: ToolRegistry,
    query: str,
    args: argparse.Namespace,
    prompt_func: Callable[[str], str] | None,
) -> None:
    initial_task_queue = None
    if state.show_plan:
        initial_task_queue = generate_action_plan(
            user_query=query,
            model_name=state.model_name,
            session_summary=store.load_summary(state.session_id),
        )
        _render_plan_with_registry(console, initial_task_queue, tool_registry)

    approval_callback = _build_approval_callback(prompt_func, console)
    task_queue = orchestrate_session(
        user_query=query,
        model_name=state.model_name,
        max_iters=args.max_iters,
        initial_task_queue=initial_task_queue,
        session_id=state.session_id,
        session_store=store,
        tool_registry=tool_registry,
        approval_callback=approval_callback,
    )

    _render_results_with_registry(console, task_queue, tool_registry)
    if task_queue.task_result:
        console.print(Panel(Markdown(task_queue.task_result), title="Response"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meeseeks terminal CLI")
    parser.add_argument("--query", help="Run a single query and exit")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--show-plan", action="store_true", default=True)
    parser.add_argument("--no-plan", action="store_false", dest="show_plan")
    parser.add_argument("--session", help="Existing session id")
    parser.add_argument("--tag", help="Session tag to resume or create")
    parser.add_argument("--fork", help="Session id or tag to fork from")
    parser.add_argument(
        "--session-dir",
        default=None,
        help="Override session storage directory",
    )
    parser.add_argument(
        "--history-file",
        default="~/.meeseeks/cli_history",
        help="Path to CLI history file",
    )
    parser.add_argument("--no-color", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())


def _tool_specs_by_id(tool_registry: ToolRegistry) -> dict[str, object]:
    return {spec.tool_id: spec for spec in tool_registry.list_specs()}


def _render_plan_with_registry(
    console: Console,
    task_queue: TaskQueue,
    tool_registry: ToolRegistry,
) -> None:
    specs = _tool_specs_by_id(tool_registry)
    table = Table(title="Action Plan", show_lines=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Action", style="magenta")
    table.add_column("Argument")
    for tool, action, argument in _format_steps(task_queue.action_steps):
        spec = specs.get(tool)
        label = tool
        if spec is not None and getattr(spec, "kind", "") == "mcp":
            label = f"{tool} (MCP)"
        table.add_row(label, action, argument)
    console.print(table)


def _render_results_with_registry(
    console: Console,
    task_queue: TaskQueue,
    tool_registry: ToolRegistry,
) -> None:
    specs = _tool_specs_by_id(tool_registry)
    table = Table(title="Tool Results", show_lines=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Result")
    for step in task_queue.action_steps:
        spec = specs.get(step.action_consumer)
        label = step.action_consumer
        if spec is not None and getattr(spec, "kind", "") == "mcp":
            label = f"{label} (MCP)"
        result = None
        if step.result is not None:
            result = getattr(step.result, "content", step.result)
        table.add_row(label, str(result))
    console.print(table)


def _build_approval_callback(
    prompt_func: Callable[[str], str] | None,
    console: Console,
) -> Callable[[ActionStep], bool] | None:
    if prompt_func is None:
        return None

    def _approve(action_step: ActionStep) -> bool:
        prompt = (
            "Approve "
            f"{action_step.action_consumer}:{action_step.action_type} "
            f"({action_step.action_argument})? [y/N] "
        )
        try:
            response = prompt_func(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\nApproval denied.")
            return False
        decision = PermissionDecision.ALLOW if response in {"y", "yes"} else PermissionDecision.DENY
        return decision == PermissionDecision.ALLOW

    return _approve

#!/usr/bin/env python3
"""Terminal CLI for Meeseeks."""

import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add repo root to path for core imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.classes import ActionStep, TaskQueue  # noqa: E402
from core.session_store import SessionStore  # noqa: E402
from core.task_master import generate_action_plan, orchestrate_session  # noqa: E402


@dataclass
class CliState:
    session_id: str
    show_plan: bool = True


COMMANDS = {
    "/help": "Show help",
    "/exit": "Exit the CLI",
    "/quit": "Exit the CLI",
    "/session": "Show current session id",
    "/summary": "Show current session summary",
    "/tag": "Tag this session: /tag NAME",
    "/fork": "Fork current session: /fork [TAG]",
    "/plan": "Toggle plan display: /plan on|off",
    "/compact": "Compact session transcript",
}


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


def _render_plan(console: Console, task_queue: TaskQueue) -> None:
    table = Table(title="Action Plan", show_lines=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Action", style="magenta")
    table.add_column("Argument")
    for tool, action, argument in _format_steps(task_queue.action_steps):
        table.add_row(tool, action, argument)
    console.print(table)


def _render_results(console: Console, task_queue: TaskQueue) -> None:
    table = Table(title="Tool Results", show_lines=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Result")
    for step in task_queue.action_steps:
        result = None
        if step.result is not None:
            result = getattr(step.result, "content", step.result)
        table.add_row(step.action_consumer, str(result))
    console.print(table)


def _print_help(console: Console) -> None:
    lines = [f"{cmd} - {desc}" for cmd, desc in COMMANDS.items()]
    console.print(Panel("\n".join(lines), title="Commands"))


def _ensure_history_path(path: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _parse_command(text: str) -> tuple[str, list[str]]:
    parts = text.strip().split()
    command = parts[0]
    return command, parts[1:]


def _handle_command(
    console: Console,
    store: SessionStore,
    state: CliState,
    command: str,
    args: list[str],
) -> bool:
    if command in {"/exit", "/quit"}:
        return False
    if command == "/help":
        _print_help(console)
        return True
    if command == "/session":
        console.print(f"Session: {state.session_id}")
        return True
    if command == "/summary":
        summary = store.load_summary(state.session_id) or "(empty)"
        console.print(Panel(summary, title="Session Summary"))
        return True
    if command == "/tag":
        if not args:
            console.print("Usage: /tag NAME")
            return True
        store.tag_session(state.session_id, args[0])
        console.print(f"Tagged session as '{args[0]}'")
        return True
    if command == "/fork":
        tag = args[0] if args else None
        new_session_id = store.fork_session(state.session_id)
        state.session_id = new_session_id
        if tag:
            store.tag_session(state.session_id, tag)
        console.print(f"Forked session: {state.session_id}")
        return True
    if command == "/plan":
        if not args:
            console.print(f"Plan display is {'on' if state.show_plan else 'off'}.")
            return True
        value = args[0].lower()
        if value in {"on", "true", "yes"}:
            state.show_plan = True
        elif value in {"off", "false", "no"}:
            state.show_plan = False
        else:
            console.print("Usage: /plan on|off")
        return True
    if command == "/compact":
        task_queue = orchestrate_session(
            "/compact",
            session_id=state.session_id,
            session_store=store,
        )
        console.print(Panel(task_queue.task_result or "", title="Compaction"))
        return True
    console.print("Unknown command. Use /help for a list of commands.")
    return True


def run_cli(args: argparse.Namespace) -> int:
    console = Console(color_system=None if args.no_color else "auto")
    store = SessionStore(root_dir=args.session_dir)
    session_id = _resolve_session_id(store, args.session, args.tag, args.fork)
    state = CliState(session_id=session_id, show_plan=args.show_plan)

    console.print(Panel("Meeseeks CLI ready", title="Meeseeks"))
    console.print(f"Session: {state.session_id}")
    console.print("Type /help for commands.\n")

    if args.query:
        return _run_single_query(console, store, state, args.query, args)

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
            if not _handle_command(console, store, state, command, cmd_args):
                return 0
            continue

        _run_query(console, store, state, user_input, args)


def _run_single_query(
    console: Console,
    store: SessionStore,
    state: CliState,
    query: str,
    args: argparse.Namespace,
) -> int:
    _run_query(console, store, state, query, args)
    return 0


def _run_query(
    console: Console,
    store: SessionStore,
    state: CliState,
    query: str,
    args: argparse.Namespace,
) -> None:
    initial_task_queue = None
    if state.show_plan:
        initial_task_queue = generate_action_plan(
            user_query=query,
            model_name=args.model,
        )
        _render_plan(console, initial_task_queue)

    task_queue = orchestrate_session(
        user_query=query,
        model_name=args.model,
        max_iters=args.max_iters,
        initial_task_queue=initial_task_queue,
        session_id=state.session_id,
        session_store=store,
    )

    _render_results(console, task_queue)
    if task_queue.task_result:
        console.print(Panel(task_queue.task_result, title="Response"))


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

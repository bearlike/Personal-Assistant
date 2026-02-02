#!/usr/bin/env python3
"""Terminal CLI for Meeseeks."""

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Add repo root to path for core imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.classes import ActionStep, TaskQueue  # noqa: E402
from core.session_store import SessionStore  # noqa: E402
from core.task_master import generate_action_plan, orchestrate_session  # noqa: E402
from core.tool_registry import ToolRegistry, load_registry  # noqa: E402


@dataclass
class CliState:
    session_id: str
    show_plan: bool = True
    model_name: str | None = None


COMMANDS = {
    "/help": "Show help",
    "/exit": "Exit the CLI",
    "/quit": "Exit the CLI",
    "/new": "Start a new session",
    "/session": "Show current session id",
    "/summary": "Show current session summary",
    "/summarize": "Summarize and compact this session",
    "/tag": "Tag this session: /tag NAME",
    "/fork": "Fork current session: /fork [TAG]",
    "/plan": "Toggle plan display: /plan on|off",
    "/compact": "Compact session transcript",
    "/mcp": "List MCP tools and servers",
    "/models": "Switch models using a local wizard",
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
    tool_registry: ToolRegistry,
    prompt_func: Callable[[str], str] | None,
) -> bool:
    if command in {"/exit", "/quit"}:
        return False
    if command == "/help":
        _print_help(console)
        return True
    if command == "/new":
        state.session_id = store.create_session()
        console.print(f"New session: {state.session_id}")
        return True
    if command == "/session":
        console.print(f"Session: {state.session_id}")
        return True
    if command == "/summary":
        summary = store.load_summary(state.session_id) or "(empty)"
        console.print(Panel(summary, title="Session Summary"))
        return True
    if command in {"/summarize", "/compact"}:
        task_queue = orchestrate_session(
            "/compact",
            session_id=state.session_id,
            session_store=store,
        )
        console.print(Panel(task_queue.task_result or "", title="Summary"))
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
    if command == "/mcp":
        _render_mcp(console, tool_registry)
        return True
    if command == "/models":
        if prompt_func is None:
            console.print("Model wizard is only available in interactive mode.")
            return True
        _handle_model_wizard(console, state, prompt_func)
        return True
    console.print("Unknown command. Use /help for a list of commands.")
    return True


def run_cli(args: argparse.Namespace) -> int:
    console = Console(color_system=None if args.no_color else "auto")
    store = SessionStore(root_dir=args.session_dir)
    session_id = _resolve_session_id(store, args.session, args.tag, args.fork)
    state = CliState(session_id=session_id, show_plan=args.show_plan, model_name=args.model)
    tool_registry = load_registry()

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
            if not _handle_command(
                console,
                store,
                state,
                command,
                cmd_args,
                tool_registry,
                session.prompt,
            ):
                return 0
            continue

        _run_query(console, store, state, tool_registry, user_input, args)


def _run_single_query(
    console: Console,
    store: SessionStore,
    state: CliState,
    tool_registry: ToolRegistry,
    query: str,
    args: argparse.Namespace,
) -> int:
    _run_query(console, store, state, tool_registry, query, args)
    return 0


def _run_query(
    console: Console,
    store: SessionStore,
    state: CliState,
    tool_registry: ToolRegistry,
    query: str,
    args: argparse.Namespace,
) -> None:
    initial_task_queue = None
    if state.show_plan:
        initial_task_queue = generate_action_plan(
            user_query=query,
            model_name=state.model_name,
            session_summary=store.load_summary(state.session_id),
        )
        _render_plan_with_registry(console, initial_task_queue, tool_registry)

    task_queue = orchestrate_session(
        user_query=query,
        model_name=state.model_name,
        max_iters=args.max_iters,
        initial_task_queue=initial_task_queue,
        session_id=state.session_id,
        session_store=store,
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


def _get_openai_base_url() -> str | None:
    return os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")


def _fetch_models() -> list[str]:
    base_url = _get_openai_base_url()
    api_key = os.getenv("OPENAI_API_KEY")
    if not base_url:
        raise RuntimeError("OPENAI_API_BASE or OPENAI_BASE_URL is not set.")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        url = f"{base_url}/models"
    else:
        url = f"{base_url}/v1/models"
    request = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Failed to fetch models: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch models: {exc.reason}") from exc
    data = payload.get("data", [])
    models = [item.get("id") for item in data if item.get("id")]
    return sorted(models)


def _handle_model_wizard(
    console: Console,
    state: CliState,
    prompt_func: Callable[[str], str],
) -> None:
    try:
        models = _fetch_models()
    except RuntimeError as exc:
        console.print(f"Model lookup failed: {exc}")
        return
    if not models:
        console.print("No models returned by the API.")
        return
    table = Table(title="Available Models", show_lines=True)
    table.add_column("Index", style="cyan")
    table.add_column("Model ID")
    for idx, model in enumerate(models, start=1):
        table.add_row(str(idx), model)
    console.print(table)
    choice = prompt_func("Select model by index or id: ").strip()
    if not choice:
        console.print("Model selection cancelled.")
        return
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(models):
            state.model_name = models[index - 1]
            console.print(f"Using model: {state.model_name}")
            return
        console.print("Invalid model index.")
        return
    if choice in models:
        state.model_name = choice
        console.print(f"Using model: {state.model_name}")
        return
    console.print("Model not recognized.")


def _render_mcp(console: Console, tool_registry: ToolRegistry) -> None:
    config_path = os.getenv("MESEEKS_MCP_CONFIG")
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as handle:
                config = json.load(handle)
            servers = config.get("servers", {})
            if servers:
                server_table = Table(title="MCP Servers", show_lines=True)
                server_table.add_column("Name", style="cyan")
                server_table.add_column("Transport")
                for name, info in servers.items():
                    server_table.add_row(name, str(info.get("transport", "")))
                console.print(server_table)
        except (json.JSONDecodeError, OSError) as exc:
            console.print(f"Failed to read MCP config: {exc}")
    specs = [spec for spec in tool_registry.list_specs() if spec.kind == "mcp"]
    if not specs:
        console.print("No MCP tools configured.")
        return
    table = Table(title="MCP Tools", show_lines=True)
    table.add_column("Tool ID", style="cyan")
    table.add_column("Server")
    table.add_column("Tool")
    for spec in specs:
        server_name = spec.metadata.get("server", "")
        tool_name = spec.metadata.get("tool", "")
        table.add_row(spec.tool_id, str(server_name), str(tool_name))
    console.print(table)

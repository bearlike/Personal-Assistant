#!/usr/bin/env python3
"""Command registry for Meeseeks CLI."""

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cli_context import CommandContext  # noqa: E402

from core.task_master import orchestrate_session  # noqa: E402
from core.token_budget import get_token_budget  # noqa: E402
from core.tool_registry import ToolRegistry  # noqa: E402


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    handler: Callable[[CommandContext, list[str]], bool]


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def command(self, name: str, help_text: str) -> Callable[[Callable], Callable]:
        def decorator(func: Callable[[CommandContext, list[str]], bool]) -> Callable:
            self._commands[name] = Command(name=name, help=help_text, handler=func)
            return func

        return decorator

    def execute(self, name: str, context: CommandContext, args: list[str]) -> bool:
        command = self._commands.get(name)
        if command is None:
            context.console.print("Unknown command. Use /help for a list of commands.")
            return True
        return command.handler(context, args)

    def help_text(self) -> str:
        lines = [f"{cmd.name} - {cmd.help}" for cmd in self._commands.values()]
        return "\n".join(sorted(lines))

    def list_commands(self) -> list[str]:
        return sorted(self._commands.keys())


REGISTRY = CommandRegistry()


@REGISTRY.command("/help", "Show help")
def _cmd_help(context: CommandContext, args: list[str]) -> bool:
    del args
    context.console.print(Panel(REGISTRY.help_text(), title="Commands"))
    return True


@REGISTRY.command("/exit", "Exit the CLI")
def _cmd_exit(context: CommandContext, args: list[str]) -> bool:
    del context, args
    return False


@REGISTRY.command("/quit", "Exit the CLI")
def _cmd_quit(context: CommandContext, args: list[str]) -> bool:
    del context, args
    return False


@REGISTRY.command("/new", "Start a new session")
def _cmd_new(context: CommandContext, args: list[str]) -> bool:
    del args
    context.state.session_id = context.store.create_session()
    context.console.print(f"New session: {context.state.session_id}")
    return True


@REGISTRY.command("/session", "Show current session id")
def _cmd_session(context: CommandContext, args: list[str]) -> bool:
    del args
    context.console.print(f"Session: {context.state.session_id}")
    return True


@REGISTRY.command("/summary", "Show current session summary")
def _cmd_summary(context: CommandContext, args: list[str]) -> bool:
    del args
    summary = context.store.load_summary(context.state.session_id) or "(empty)"
    context.console.print(Panel(summary, title="Session Summary"))
    return True


@REGISTRY.command("/summarize", "Summarize and compact this session")
def _cmd_summarize(context: CommandContext, args: list[str]) -> bool:
    del args
    task_queue = orchestrate_session(
        "/compact",
        session_id=context.state.session_id,
        session_store=context.store,
    )
    context.console.print(Panel(task_queue.task_result or "", title="Summary"))
    return True


@REGISTRY.command("/compact", "Compact session transcript")
def _cmd_compact(context: CommandContext, args: list[str]) -> bool:
    return _cmd_summarize(context, args)


@REGISTRY.command("/tag", "Tag this session: /tag NAME")
def _cmd_tag(context: CommandContext, args: list[str]) -> bool:
    if not args:
        context.console.print("Usage: /tag NAME")
        return True
    context.store.tag_session(context.state.session_id, args[0])
    context.console.print(f"Tagged session as '{args[0]}'")
    return True


@REGISTRY.command("/fork", "Fork current session: /fork [TAG]")
def _cmd_fork(context: CommandContext, args: list[str]) -> bool:
    tag = args[0] if args else None
    new_session_id = context.store.fork_session(context.state.session_id)
    context.state.session_id = new_session_id
    if tag:
        context.store.tag_session(context.state.session_id, tag)
    context.console.print(f"Forked session: {context.state.session_id}")
    return True


@REGISTRY.command("/plan", "Toggle plan display: /plan on|off")
def _cmd_plan(context: CommandContext, args: list[str]) -> bool:
    if not args:
        context.console.print(
            f"Plan display is {'on' if context.state.show_plan else 'off'}."
        )
        return True
    value = args[0].lower()
    if value in {"on", "true", "yes"}:
        context.state.show_plan = True
    elif value in {"off", "false", "no"}:
        context.state.show_plan = False
    else:
        context.console.print("Usage: /plan on|off")
    return True


@REGISTRY.command("/mcp", "List MCP tools and servers")
def _cmd_mcp(context: CommandContext, args: list[str]) -> bool:
    del args
    _render_mcp(context.console, context.tool_registry)
    return True


@REGISTRY.command("/models", "Switch models using a local wizard")
def _cmd_models(context: CommandContext, args: list[str]) -> bool:
    del args
    if context.prompt_func is None:
        context.console.print("Model wizard is only available in interactive mode.")
        return True
    _handle_model_wizard(context.console, context, context.prompt_func)
    return True


@REGISTRY.command("/tokens", "Show token usage and remaining context")
def _cmd_tokens(context: CommandContext, args: list[str]) -> bool:
    del args
    events = context.store.load_transcript(context.state.session_id)
    summary = context.store.load_summary(context.state.session_id)
    budget = get_token_budget(events, summary, context.state.model_name)
    table = Table(title="Token Budget", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Summary tokens", str(budget.summary_tokens))
    table.add_row("Event tokens", str(budget.event_tokens))
    table.add_row("Total tokens", str(budget.total_tokens))
    table.add_row("Context window", str(budget.context_window))
    table.add_row("Remaining", str(budget.remaining_tokens))
    table.add_row("Utilization", f"{budget.utilization:.1%}")
    table.add_row("Auto-compact threshold", f"{budget.threshold:.1%}")
    context.console.print(table)
    return True


@REGISTRY.command("/budget", "Show token usage and remaining context")
def _cmd_budget(context: CommandContext, args: list[str]) -> bool:
    return _cmd_tokens(context, args)


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
    context: CommandContext,
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
            context.state.model_name = models[index - 1]
            console.print(f"Using model: {context.state.model_name}")
            return
        console.print("Invalid model index.")
        return
    if choice in models:
        context.state.model_name = choice
        console.print(f"Using model: {context.state.model_name}")
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


def get_registry() -> CommandRegistry:
    return REGISTRY


__all__ = ["CommandContext", "CommandRegistry", "get_registry"]

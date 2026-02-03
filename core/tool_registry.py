#!/usr/bin/env python3
"""Tool registry and manifest loading for Meeseeks."""
from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.classes import ActionStep, set_available_tools
from core.common import MockSpeaker, get_logger

logging = get_logger(name="core.tool_registry")


class ToolRunner(Protocol):
    def run(self, action_step: ActionStep) -> MockSpeaker:  # pragma: no cover
        """Execute an action step and return a speaker response."""


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    name: str
    description: str
    factory: Callable[[], ToolRunner]
    enabled: bool = True
    kind: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._instances: dict[str, ToolRunner] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.tool_id] = spec
        set_available_tools(list(self._tools.keys()))

    def get(self, tool_id: str) -> ToolRunner | None:
        spec = self._tools.get(tool_id)
        if spec is None or not spec.enabled:
            return None
        if tool_id not in self._instances:
            self._instances[tool_id] = spec.factory()
        return self._instances[tool_id]

    def list_specs(self, include_disabled: bool = False) -> list[ToolSpec]:
        specs = list(self._tools.values())
        if include_disabled:
            return specs
        return [spec for spec in specs if spec.enabled]

    def tool_catalog(self) -> list[dict[str, str]]:
        return [
            {
                "tool_id": spec.tool_id,
                "name": spec.name,
                "description": spec.description,
            }
            for spec in self.list_specs()
        ]


def _import_factory(module_path: str, class_name: str) -> Callable[[], ToolRunner]:
    def _factory() -> ToolRunner:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    return _factory


def _default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            tool_id="home_assistant_tool",
            name="Home Assistant",
            description="Manage smart home devices via Home Assistant.",
            factory=_import_factory(
                "tools.integration.homeassistant",
                "HomeAssistant",
            ),
        )
    )
    registry.register(
        ToolSpec(
            tool_id="talk_to_user_tool",
            name="Talk to User",
            description="Respond directly to the user.",
            factory=_import_factory(
                "tools.core.talk_to_user",
                "TalkToUser",
            ),
        )
    )
    return registry


def load_registry(manifest_path: str | None = None) -> ToolRegistry:
    """Load tool registry from a JSON manifest if available.

    Manifest format:
    {
      "tools": [
        {"tool_id": "...", "name": "...", "description": "...", "module": "...",
         "class": "...", "enabled": true, "kind": "local"},
        {"tool_id": "mcp_weather", "name": "Weather", "description": "...",
         "kind": "mcp", "server": "weather", "tool": "get_weather"}
      ]
    }
    """
    if manifest_path is None:
        manifest_path = os.getenv("MESEEKS_TOOL_MANIFEST")

    if not manifest_path:
        return _default_registry()

    manifest_path = os.path.abspath(manifest_path)
    if not os.path.exists(manifest_path):
        logging.warning("Tool manifest not found: %s", manifest_path)
        return _default_registry()

    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("Failed to load tool manifest: %s", exc)
        return _default_registry()

    registry = ToolRegistry()
    for tool in manifest.get("tools", []):
        kind = tool.get("kind", "local")
        if kind == "local":
            module_path = tool.get("module")
            class_name = tool.get("class")
            if not module_path or not class_name:
                logging.warning("Skipping tool with missing module/class: %s", tool)
                continue
            factory = _import_factory(module_path, class_name)
        else:
            from tools.integration.mcp import MCPToolRunner

            server_name = tool.get("server")
            tool_name = tool.get("tool")
            if not server_name or not tool_name:
                logging.warning("Skipping MCP tool with missing server/tool: %s", tool)
                continue
            def _mcp_factory(
                server_name: str = server_name,
                tool_name: str = tool_name,
            ) -> MCPToolRunner:
                return MCPToolRunner(server_name=server_name, tool_name=tool_name)

            factory = _mcp_factory

        spec = ToolSpec(
            tool_id=tool.get("tool_id", ""),
            name=tool.get("name", tool.get("tool_id", "")),
            description=tool.get("description", ""),
            factory=factory,
            enabled=tool.get("enabled", True),
            kind=kind,
            metadata={
                key: value
                for key, value in tool.items()
                if key
                not in {
                    "tool_id",
                    "name",
                    "description",
                    "module",
                    "class",
                    "enabled",
                    "kind",
                }
            },
        )
        if not spec.tool_id:
            logging.warning("Skipping tool with empty tool_id: %s", tool)
            continue
        registry.register(spec)

    if not registry.list_specs(include_disabled=True):
        return _default_registry()

    return registry


__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "load_registry",
]

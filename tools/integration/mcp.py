#!/usr/bin/env python3
"""MCP tool runner for integrating MCP servers into Meeseeks."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from core.common import MockSpeaker, get_logger, get_mock_speaker

logging = get_logger(name="tools.integration.mcp")


def _load_mcp_config() -> dict[str, Any]:
    config_path = os.getenv("MESEEKS_MCP_CONFIG")
    if not config_path:
        raise ValueError("MESEEKS_MCP_CONFIG is not set.")
    config_path = os.path.abspath(config_path)
    with open(config_path, encoding="utf-8") as handle:
        return json.load(handle)


class MCPToolRunner:
    """Wrapper to invoke MCP tools via langchain-mcp-adapters."""

    def __init__(self, server_name: str, tool_name: str) -> None:
        self.server_name = server_name
        self.tool_name = tool_name

    async def _invoke_async(self, input_text: str) -> str:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "langchain-mcp-adapters is required for MCP tools."
            ) from exc

        config = _load_mcp_config()
        servers = config.get("servers", {})
        if not servers or self.server_name not in servers:
            raise ValueError(
                f"MCP server '{self.server_name}' not found in config."
            )

        client = MultiServerMCPClient({self.server_name: servers[self.server_name]})
        tools = await client.get_tools(server_name=self.server_name)
        tool_map = {tool.name: tool for tool in tools}
        tool = tool_map.get(self.tool_name)
        if tool is None:
            raise ValueError(
                f"Tool '{self.tool_name}' not found on MCP server '{self.server_name}'."
            )
        result = await tool.ainvoke(input_text)
        return str(result)

    def run(self, action_step) -> MockSpeaker:
        if action_step is None:
            raise ValueError("Action step cannot be None.")
        MockSpeakerType = get_mock_speaker()
        result = asyncio.run(self._invoke_async(action_step.action_argument))
        return MockSpeakerType(content=result)


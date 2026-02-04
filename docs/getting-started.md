# Getting Started

This guide walks through local setup, environment configuration, MCP setup, and how to run each interface.

## Prerequisites
- Python 3.10+
- uv
- Docker (optional, for container runs)

## Install dependencies
```bash
uv venv .venv
uv pip install -e .[dev]
uv pip install -e packages/meeseeks_core -e packages/meeseeks_tools \
  -e apps/meeseeks_api -e apps/meeseeks_chat -e apps/meeseeks_cli \
  -e meeseeks_ha_conversation
```

## Environment setup
1. Copy `.env.example` to `.env`.
2. Set at least:
   - `OPENAI_API_KEY` (for your OpenAI-compatible endpoint)
   - `OPENAI_API_BASE` (LiteLLM proxy or other OpenAI-compatible base URL)
   - `DEFAULT_MODEL` (or `ACTION_PLAN_MODEL`)
3. If you use an OpenAI-compatible base URL and your model name has no provider
   prefix, Meeseeks will call `openai/<model>` automatically.
3. Optional runtime paths:
   - `MESEEKS_SESSION_DIR` for session transcript storage
   - `MESEEKS_TOOL_MANIFEST` if you want a custom tool list (disables MCP auto-discovery)

## MCP setup (auto-discovery)
MCP tools are auto-discovered from a server config file.
1. Copy `configs/mcp.example.json` to `configs/mcp.json`.
2. Set the MCP server `url` and any `headers` needed for auth.
3. Set `MESEEKS_MCP_CONFIG=./configs/mcp.json` in `.env`.
4. Start any interface once; a tool manifest is auto-generated and cached under `~/.meeseeks/`.

Notes:
If you override the manifest, keep at least one tool enabled for tasks that need external actions.
- MCP tool names must match the server's advertised tool list.

## Optional components
- Langfuse: set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` (or disable with `LANGFUSE_ENABLED=0`).
- Home Assistant: set `HA_URL` + `HA_TOKEN` (or disable with `MESEEKS_HOME_ASSISTANT_ENABLED=0`).

## Run interfaces (local)
- API: `meeseeks-api` (or `python -m meeseeks_api.backend`)
- Chat UI: `meeseeks-chat`
- CLI: `meeseeks`
- Home Assistant integration: install `meeseeks_ha_conversation/` as a custom component and point it at the API.

## Docker (optional)
- Build images using `docker/Dockerfile.api` and `docker/Dockerfile.chat`.
- Provide the same `.env` values as local.
- Persist `MESEEKS_SESSION_DIR` if you want transcripts across restarts.

## Docs (optional)
If you want to build the docs locally:
```bash
uv venv .venv
uv pip install -e .[docs] -e packages/meeseeks_core -e packages/meeseeks_tools \
  -e meeseeks_ha_conversation
mkdocs serve
```

# CLI Client

The CLI runs the session runtime in-process and is the fastest way to validate local setup.

## Setup (uv)
```bash
uv sync --extra cli
```

Confirm LLM configuration first: see `llm-setup.md`.

## Run
```bash
uv run meeseeks
```

## Common flags
- `--query "..."` run a single request and exit.
- `--model MODEL_NAME` override the configured model.
- `--session SESSION_ID` resume a session by id.
- `--tag TAG` resume or create a tagged session.
- `--session-dir PATH` override transcript storage.
- `--auto-approve` auto-approve tool permissions for the session.

## Notes
- Core commands are available: `/compact`, `/status`, `/terminate`.
- MCP tools are loaded from `configs/mcp.json` when present.

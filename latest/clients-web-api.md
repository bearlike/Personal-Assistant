# Web + API Clients

This page covers the Streamlit chat UI and the REST API. Both use the same core engine.

## Setup (uv)
```bash
uv sync --extra api --extra chat
```

Confirm LLM configuration first: see `llm-setup.md`.

## Run the REST API
```bash
uv run meeseeks-api
```

API notes:
- Protected routes require `X-API-Key` matching `api.master_token` in `configs/app.json`.
- Session runtime endpoints support async runs and event polling.

## Run the web UI (Streamlit)
```bash
uv run meeseeks-chat
```

UI notes:
- Streamlit options live under `chat.*` in `configs/app.json` (address and port).
- The chat UI runs the core engine in-process.

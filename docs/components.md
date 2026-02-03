# Components

This repository is a monorepo. Each component lives in its own folder:

- `core/`: orchestration loop, schemas, session storage, compaction, tool registry.
- `tools/`: tool implementations and integration glue.
- `meeseeks-api/`: Flask API that exposes the assistant over HTTP.
- `meeseeks-chat/`: Streamlit UI for interactive chat.
- `meeseeks-cli/`: terminal CLI for interactive sessions.
- `meeseeks_ha_conversation/`: Home Assistant integration that routes voice requests to the API.
- `prompts/`: planner prompt and examples.

Note: the API/Chat/CLI folders use hyphens in their names, which makes them invalid Python module names. MkDocs + mkdocstrings can only render inline docstrings for importable modules, so those folders will not show API docs until we add a package shim or rename them.

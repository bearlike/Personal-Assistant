# Home Assistant Voice (HA Assist)

The Home Assistant integration connects Assist to the Meeseeks REST API for voice requests.

## Setup (uv)
```bash
uv sync --extra api --extra ha
```

Confirm LLM configuration first: see `llm-setup.md`.

## Install the custom component
1. Run the Meeseeks API: `uv run meeseeks-api`.
2. Copy the contents of `meeseeks_ha_conversation/` into Home Assistant under
   `custom_components/meeseeks_conversation/`.
3. In Home Assistant, add the "Meeseeks" conversation integration and set:
   - Base URL: the API base URL (for example, `http://host:5123`).
   - API key: the API master token (`api.master_token` in `configs/app.json`).

## Optional: enable the Home Assistant tool
If Meeseeks should control Home Assistant entities directly:
- Set `home_assistant.enabled` to `true` in `configs/app.json`.
- Provide the Home Assistant URL and token in `home_assistant.*`.

# LLM Core Setup

This page covers the minimum LLM configuration required to run the core engine.

## Minimum configuration
Set these keys in `configs/app.json`:

```json
{
  "llm": {
    "api_base": "https://your-llm-endpoint/v1",
    "api_key": "sk-your-key",
    "default_model": "gpt-5.2"
  }
}
```

Optional (recommended for separation of concerns):
- `llm.action_plan_model`: model for plan generation.
- `llm.tool_model`: model for tool execution.
- `llm.reasoning_effort` and `llm.reasoning_effort_models`: enable reasoning effort where supported.

## Short walkthrough
1. Copy the example config: `cp configs/app.example.json configs/app.json`.
2. Edit the `llm` block with the API base, API key, and model names.
3. Start a client (CLI, API, or chat). See the client pages for run commands.

## LiteLLM provider support
The LLM layer is backed by LiteLLM via `langchain-litellm`.

- Model names can include provider prefixes (for example, `openai/gpt-4-turbo`, `anthropic/claude-3-sonnet`, or `mistral/mistral-small`).
- If `llm.api_base` is set and a model has no provider prefix, the system defaults to `openai/<model>` to match OpenAI-style endpoints.
- For multi-provider routing, see `docs/litellm-config.example.yml`.

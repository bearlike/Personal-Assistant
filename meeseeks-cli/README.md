# Meeseeks CLI

A terminal frontend for Meeseeks. It runs the same orchestration loop as the API and chat UI, but in a fast, interactive shell.

## Features
- Interactive conversations in the terminal.
- Shows action plans and tool results per request.
- Session transcripts and compaction from the core engine.
- Tag and fork sessions for experiments.

## Run
```bash
poetry run python cli_master.py
```

## Common commands
- `/help` list commands
- `/plan on|off` toggle plan display
- `/summarize` compact the session transcript
- `/tag NAME` tag a session
- `/fork [TAG]` fork the current session
- `/new` start a new session
- `/mcp` list MCP tools and servers
- `/models` switch models using a wizard
- `/quit` exit the CLI

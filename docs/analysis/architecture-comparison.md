# Architecture Comparison: Meeseeks vs. Claude Code

## Scope
This document summarizes the architectural blocks in the Meeseeks Personal Assistant codebase and compares them with Claude Code patterns from DeepWiki. It focuses on orchestration, tool execution, extensibility, and context management so we can identify gaps and plan targeted fixes.

## Meeseeks (Personal Assistant) Architecture Blocks

### Entry Points / Interfaces
- **REST API**: Flask-RESTX endpoint `/api/query` accepts a user query, generates an action plan, executes it, and returns the `TaskQueue` payload as JSON. The API enforces an API token before execution. (Files: `meeseeks-api/backend.py`.)
- **Chat UI**: Streamlit chat collects user prompts, displays the action plan, executes the plan, and renders the aggregated response. (Files: `meeseeks-chat/chat_master.py`.)
- **Home Assistant integration**: The custom HA conversation integration routes voice requests to the API (documented externally in the HA integration package, but the core pipeline is the same API/TaskMaster flow).

### Planning & Orchestration
- **Planner**: `generate_action_plan()` uses a system prompt + few-shot examples + a Pydantic output parser (`TaskQueue`) to structure the plan into `ActionStep` objects. It logs tracing metadata using Langfuse. (Files: `core/task_master.py`.)
- **Executor**: `run_action_plan()` maps each `ActionStep` to a tool instance, runs the tool, and aggregates tool outputs into `task_result`. (Files: `core/task_master.py`.)

### Schema & Validation
- **ActionStep / TaskQueue** enforce allowed tool IDs (`home_assistant_tool`, `talk_to_user_tool`), allowed action types (`get`/`set`), and non-null arguments. (Files: `core/classes.py`.)

### Tool Framework
- **AbstractTool** handles model configuration, Langfuse tracing, caching, and `get`/`set` execution routing. (Files: `core/classes.py`.)
- **HomeAssistant tool** builds a cache of entities/services, renders prompts for set/get flows, performs LLM-based structured calls for `set`, and summarises sensor context for `get`. (Files: `tools/integration/homeassistant.py`.)
- **TalkToUser tool** returns a message verbatim to speak to the user and does not implement `get`. (Files: `tools/core/talk_to_user.py`.)

### Prompting Policy
- The action planner prompt defines valid tools, action types, and guidelines (e.g., avoid `talk_to_user_tool` after `get`). (Files: `prompts/action-planner.txt`.)

## Claude Code Architectural Patterns (DeepWiki)

The DeepWiki architecture overview for Claude Code describes a layered system design with more extensive orchestration, extensibility, and context management:

### Orchestration & Agents
- A **main agent** handles the user’s interactive thread, while **subagents** can be spawned for parallel/background tasks. A dedicated **plan agent** is used for multi-step planning in “plan mode.”
- Task tools can spawn subagents with model/tool restrictions and report results back to the main agent.

### Tool Execution & Permissions
- Built-in tools include Bash, Read/Write/Search, LSP, Task, and AskUserQuestion, all protected by a **permission system** with wildcard rules (e.g., `Write(docs/*.md)`).
- A permission dialog is triggered when no rule matches, and **PermissionRequest hooks** can auto-approve or deny requests.

### Hooks & Lifecycle Interception
- A hook system provides `PreToolUse`, `PostToolUse`, `SessionStart`, `Stop`, `PermissionRequest`, and other events. Hooks can inject context or modify tool input/output.

### Context Management / Compaction
- Session transcripts are persisted, and **auto-compaction** triggers at ~98% context utilization. The compaction result becomes the active summary when a session resumes. MCP tool descriptions can be deferred to reduce context load.

### Extensibility (Plugins, Skills, MCP)
- Plugins expose commands, agents, skills, hooks, and optional MCP server definitions.
- MCP supports multiple transports (stdio/HTTP/SSE/OAuth) and can defer tool discovery through an `MCPSearch` mechanism when tool descriptions become too large.

### Framework Options for MCP Integration (external references)
- **LangChain MCP adapters**: provide adapter functions that convert MCP tools/prompts/resources into LangChain-native types and support a multi-server MCP client with stdio/HTTP/SSE/WebSocket transports (see the `langchain-mcp-adapters` DeepWiki overview).  
- **PydanticAI MCP tools**: the PydanticAI docs list MCP tools under third-party integrations; these can be used for a Pydantic-first integration path alongside LangChain tools.

## Aider (External Reference) Orchestration Signals
- Aider is a Python CLI that sustains long-running coding sessions and includes built-in chat history handling plus summarization utilities for large histories (useful for session artifacts/compaction design).
- The CLI runs a persistent loop around its core coder, showing how a minimal controller loop can keep the session alive without constant re-parsing of user prompts.

## Observed Gaps / Missing Mechanisms in Meeseeks

### 1) Orchestration Depth
- **Current**: single-shot plan → execute → aggregate.
- **Missing**: plan-mode or iterative planning; subagent spawning; background task management; re-planning on tool failures.

### 2) Tool Governance & Permissions
- **Current**: no explicit permission gating; the tool registry is hard-coded and always available.
- **Missing**: rule-based permissions, human-in-the-loop approval, hook-based interception before tool use.

### 3) Context & Session Management
- **Current**: per-request stateless planner; Streamlit maintains a short in-memory buffer, but core doesn’t persist session transcripts or compact history.
- **Missing**: transcript persistence, compaction rules, context budgets, and resume/fork operations.

### 4) Extensibility
- **Current**: tools are manually registered; no plugin/skill/hook discovery.
- **Missing**: plugin manifests, hook/skill frontmatter, MCP configuration lifecycle, marketplace or project-local tool catalogs.

### 5) Functional Reliability Patterns
- **Current**: errors are logged per step but not used to drive adaptive behavior.
- **Missing**: automated retries, tool fallback strategies, confidence thresholds, or tool-specific error remediation flows.

## Targeted Fix Path (KISS/DRY)

### Phase 1: Core Orchestration Upgrade (single agent → looped controller)
**Goal:** replace single-shot planning with a minimal controller loop that preserves task state across turns and keeps the model grounded in tool results.

**Inspiration (external):** Aider maintains chat history files and has a summarization helper for long histories, and its CLI runs a continuous loop that repeatedly invokes the core coder until completion. These patterns are useful for long-running workflows and for preventing loss of task context.

**Controller loop contract (minimal surface):**
- **Inputs:** user request, prior loop state (if any), current session summary.
- **Loop phases:** `plan → act → observe → decide`.
- **Outputs:** updated loop state, tool results, optional user-facing response, and a completion flag.

**Proposed loop state (lightweight JSON structure):**
- `goal`: normalized user intent.
- `plan`: current ordered list of `ActionStep` tasks (can be revised).
- `tool_results`: last N tool outputs with timestamps.
- `open_questions`: clarifications needed for progress.
- `done`: boolean + reason (success, blocked, needs user input).
- `summary`: rolling summary used for compaction.

**Key behaviors to implement (KISS):**
1. **Action plan can be revised** after tool results (e.g., if a tool fails or reveals new constraints).
2. **Decision gate** determines whether to continue looping, ask user, or terminate.
3. **Observation step** is just “summarize last tool output + update state.”
4. **Short-term memory** is the loop state; **long-term memory** is the session transcript (Phase 3).

**Minimal code changes to start (order of operations):**
1. Wrap `generate_action_plan()` + `run_action_plan()` inside a new `orchestrate_session()` loop (in `core/task_master.py`).
2. Persist loop state in memory for chat/API sessions (later to disk).
3. Add a small completion check (e.g., `done` if no remaining steps or user question answered).
4. Keep `ActionStep` + `TaskQueue` schema as-is, but allow re-generation of `TaskQueue` when needed.

**Why this is the most leverage:** the looped controller is the core enabler for long-running, coherent conversations, and it unblocks later improvements (MCP tools, session artifacts, compaction) without major schema changes.

### Phase 2: MCP Integration (minimal surface, maximum leverage)
- Add MCP client support using **LangChain MCP adapters** (bridge MCP tools/prompts/resources into LangChain `BaseTool`/messages). This gives us tool discovery across multiple servers with minimal local code.
- Optionally, add **PydanticAI MCP client** as an alternative path when a Pydantic-first stack is preferred (MCP tooling is first-class in PydanticAI docs and can coexist with LangChain-based orchestration).
- Start with a **single MCP server** configured via a lightweight JSON config to prove the integration before expanding.

### Phase 3: Session Artifacts & Compaction
- Persist **transcripts (JSONL)** per session and add a manual `/compact` command.
- Add **auto-compaction** once transcripts exist to keep long-running sessions within model context (mirrors Claude Code’s approach).

### Phase 4: Session Management (resume/fork/branch)
- Support **resume** (load transcripts + compact summary), **fork** (branch a session for experimentation), and **tagging** (for session lookup).
- Keep the implementation small: a session index file + transcript store.

### Phase 5: Extensibility Entry Point
- Add a basic registry/manifest for tools so new MCP tools or local tools can be enabled without code edits. Avoid a marketplace at first.

## Next Questions to Drive Implementation
1. Should Meeseeks prioritize **parallel subagents** or a **single-agent iterative plan/act loop** first?
2. What are the **minimum permissions** that must require human confirmation (e.g., `set` vs. `get`)?
3. Do we need **session persistence** across chat and API immediately, or only for chat?
4. Should tool discovery stay **static** for now, or move to a **manifest-driven registry** in Phase 4?

## Implementation Status (Feb 2, 2026)
- Phase 1: Loop-based orchestration is live (`core/task_master.py`) with replanning and state tracking.
- Phase 2: MCP tool runner + manifest wiring added (`tools/integration/mcp.py`, `core/tool_registry.py`).
- Phase 3: Transcript persistence + compaction helpers wired in (`core/session_store.py`, `core/compaction.py`).
- Phase 4: Session fork/tag/resume support in API/chat (`meeseeks-api/backend.py`, `meeseeks-chat/chat_master.py`).
- Phase 5: Tool registry/manifest entry point in place for local/MCP tools (`core/tool_registry.py`).

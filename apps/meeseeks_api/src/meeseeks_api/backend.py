#!/usr/bin/env python3
"""Meeseeks API.

Single-user REST API with session-based orchestration and event polling.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import threading

from flask import Flask, request
from flask_restx import Api, Resource, fields

from meeseeks_core.classes import TaskQueue
from meeseeks_core.common import get_logger
from meeseeks_core.config import get_config, get_config_value, start_preflight
from meeseeks_core.permissions import auto_approve
from meeseeks_core.session_store import SessionStore
from meeseeks_core.task_master import orchestrate_session

# Get the API token from app config
MASTER_API_TOKEN = get_config_value("api", "master_token", default="msk-strong-password")

# Initialize logger
logging = get_logger(name="meeseeks-api")
logging.info("Starting Meeseeks API server.")
logging.debug("Starting API server with API token: {}", MASTER_API_TOKEN)

_config = get_config()
if _config.runtime.preflight_enabled:
    start_preflight(_config)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class RunHandle:
    """Active orchestration thread tracking."""

    thread: threading.Thread
    cancel_event: threading.Event
    started_at: str


class RunRegistry:
    """Track active orchestration threads per session."""

    def __init__(self) -> None:
        """Initialize the run registry."""
        self._lock = threading.Lock()
        self._runs: dict[str, RunHandle] = {}

    def start(
        self,
        session_id: str,
        target: Callable[[threading.Event], None],
    ) -> bool:
        """Start a new run for the session if one is not already active."""
        with self._lock:
            existing = self._runs.get(session_id)
            if existing and existing.thread.is_alive():
                return False
            cancel_event = threading.Event()
            thread = threading.Thread(
                target=self._wrap_run,
                args=(session_id, cancel_event, target),
                daemon=True,
            )
            self._runs[session_id] = RunHandle(
                thread=thread,
                cancel_event=cancel_event,
                started_at=_utc_now(),
            )
            thread.start()
            return True

    def _wrap_run(
        self,
        session_id: str,
        cancel_event: threading.Event,
        target: Callable[[threading.Event], None],
    ) -> None:
        try:
            target(cancel_event)
        finally:
            with self._lock:
                handle = self._runs.get(session_id)
                if handle and handle.thread.ident == threading.current_thread().ident:
                    self._runs.pop(session_id, None)

    def cancel(self, session_id: str) -> bool:
        """Request cancellation for an active session run."""
        with self._lock:
            handle = self._runs.get(session_id)
            if not handle:
                return False
            handle.cancel_event.set()
            return True

    def is_running(self, session_id: str) -> bool:
        """Return True if the session has an active run."""
        with self._lock:
            handle = self._runs.get(session_id)
            return bool(handle and handle.thread.is_alive())

    def get_cancel_event(self, session_id: str) -> threading.Event | None:
        """Return the cancel event for a session, if present."""
        with self._lock:
            handle = self._runs.get(session_id)
            return handle.cancel_event if handle else None


# Create Flask application
app = Flask(__name__)
session_store = SessionStore()
run_registry = RunRegistry()

authorizations = {"apikey": {"type": "apiKey", "in": "header", "name": "X-API-KEY"}}
VERSION = get_config_value("runtime", "version", default="(Dev)")
api = Api(
    app,
    version=VERSION,
    title="Meeseeks API",
    description="Interact with Meeseeks through a REST API",
    doc="/swagger-ui/",
    authorizations=authorizations,
    security="apikey",
)

ns = api.namespace("api", description="Meeseeks operations")

task_queue_model = api.model(
    "TaskQueue",
    {
        "session_id": fields.String(
            required=False, description="Session identifier for transcript storage"
        ),
        "human_message": fields.String(required=True, description="The original user query"),
        "task_result": fields.String(
            required=True, description="Combined response of all action steps"
        ),
        "action_steps": fields.List(
            fields.Nested(
                api.model(
                    "ActionStep",
                    {
                        "action_consumer": fields.String(
                            required=True,
                            description="The tool responsible for executing the action",
                        ),
                        "action_type": fields.String(
                            required=True,
                            description="The type of action to be performed (get/set)",
                        ),
                        "action_argument": fields.String(
                            required=True, description="The specific argument for the action"
                        ),
                        "result": fields.String(description="The result of the executed action"),
                    },
                )
            )
        ),
    },
)


@app.before_request
def log_request_info() -> None:
    """Log request metadata for debugging."""
    logging.debug("Endpoint: {}", request.endpoint)
    logging.debug("Headers: {}", request.headers)
    logging.debug("Body: {}", request.get_data())


def _require_api_key() -> tuple[dict, int] | None:
    api_token = request.headers.get("X-API-Key", None)
    if api_token is None:
        return {"message": "API token is not provided."}, 401
    if api_token != MASTER_API_TOKEN:
        logging.warning("Unauthorized API call attempt with token: {}", api_token)
        return {"message": "Unauthorized"}, 401
    return None


def _append_context_event(session_id: str, context: dict[str, object]) -> None:
    if not context:
        return
    session_store.append_event(session_id, {"type": "context", "payload": context})


def _summarize_session(session_id: str) -> dict[str, object]:
    events = session_store.load_transcript(session_id)
    created_at = events[0]["ts"] if events else None
    title = None
    status = "idle"
    done_reason = None
    context: dict[str, object] | None = None
    for event in events:
        if event.get("type") == "context":
            payload = event.get("payload")
            if isinstance(payload, dict):
                context = payload
        if title is None and event.get("type") == "user":
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                title = payload.get("text")
        if event.get("type") == "completion":
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                done_reason = payload.get("done_reason")
                status = "completed" if payload.get("done") else "incomplete"
    running = run_registry.is_running(session_id)
    if running:
        status = "running"
    if not title:
        title = f"Session {session_id[:8]}"
    return {
        "session_id": session_id,
        "title": title,
        "created_at": created_at,
        "status": status,
        "done_reason": done_reason,
        "running": running,
        "context": context or {},
    }


def _filter_events(events: list[dict], after_ts: str | None) -> list[dict]:
    if not after_ts:
        return events
    cutoff = _parse_iso(after_ts)
    if cutoff is None:
        return events
    filtered: list[dict] = []
    for event in events:
        ts = _parse_iso(event.get("ts"))
        if ts and ts > cutoff:
            filtered.append(event)
    return filtered


def _handle_slash_command(session_id: str, user_query: str) -> tuple[dict, int] | None:
    command = user_query.strip().lower().split()[0]
    if command in {"/terminate", "/quit", "/cancel", "/stop"}:
        canceled = run_registry.cancel(session_id)
        return {"session_id": session_id, "canceled": canceled}, 202
    if command == "/status":
        return {"session_id": session_id, **_summarize_session(session_id)}, 200
    if command == "/compact":
        # Let orchestrator handle compaction; it will write summary and completion.
        return None
    return None


def _run_orchestration(
    session_id: str,
    user_query: str,
    cancel_event: threading.Event,
) -> None:
    orchestrate_session(
        user_query=user_query,
        session_id=session_id,
        session_store=session_store,
        approval_callback=auto_approve,
        should_cancel=cancel_event.is_set,
    )


@ns.route("/sessions")
class Sessions(Resource):
    """List and create sessions."""

    @api.doc(security="apikey")
    def get(self) -> tuple[dict, int]:
        """List sessions for the single user."""
        auth_error = _require_api_key()
        if auth_error:
            return auth_error
        sessions = [_summarize_session(sid) for sid in session_store.list_sessions()]
        return {"sessions": sessions}, 200

    @api.doc(security="apikey")
    def post(self) -> tuple[dict, int]:
        """Create a new session."""
        auth_error = _require_api_key()
        if auth_error:
            return auth_error
        payload = request.get_json(silent=True) or {}
        session_id = session_store.create_session()
        session_tag = payload.get("session_tag")
        if session_tag:
            session_store.tag_session(session_id, session_tag)
        context = payload.get("context")
        if isinstance(context, dict):
            _append_context_event(session_id, context)
        return {"session_id": session_id}, 200


@ns.route("/sessions/<string:session_id>/query")
class SessionQuery(Resource):
    """Enqueue a query or process slash commands for a session."""

    @api.doc(security="apikey")
    def post(self, session_id: str) -> tuple[dict, int]:
        """Handle a session query."""
        auth_error = _require_api_key()
        if auth_error:
            return auth_error
        request_data = request.get_json(silent=True) or {}
        user_query = request_data.get("query")
        if not user_query:
            return {"message": "Invalid input: 'query' is required"}, 400

        command_response = _handle_slash_command(session_id, user_query)
        if command_response is not None:
            return command_response

        if run_registry.is_running(session_id):
            return {"message": "Session is already running."}, 409

        context = request_data.get("context")
        if isinstance(context, dict):
            _append_context_event(session_id, context)

        started = run_registry.start(
            session_id,
            target=lambda cancel_event: _run_orchestration(session_id, user_query, cancel_event),
        )
        if not started:
            return {"message": "Session is already running."}, 409
        return {"session_id": session_id, "accepted": True}, 202


@ns.route("/sessions/<string:session_id>/events")
class SessionEvents(Resource):
    """Return session events for polling."""

    @api.doc(security="apikey")
    def get(self, session_id: str) -> tuple[dict, int]:
        """Return events for the session."""
        auth_error = _require_api_key()
        if auth_error:
            return auth_error
        after_ts = request.args.get("after")
        events = session_store.load_transcript(session_id)
        events = _filter_events(events, after_ts)
        return {
            "session_id": session_id,
            "events": events,
            "running": run_registry.is_running(session_id),
        }, 200


@ns.route("/query")
class MeeseeksQuery(Resource):
    """Legacy sync endpoint (CLI compatibility)."""

    @api.doc(security="apikey")
    @api.expect(
        api.model(
            "Query",
            {
                "query": fields.String(required=True, description="The user query"),
                "session_id": fields.String(required=False, description="Existing session id"),
                "session_tag": fields.String(required=False, description="Human-friendly tag"),
                "fork_from": fields.String(required=False, description="Session id or tag to fork"),
            },
        )
    )
    @api.response(200, "Success", task_queue_model)
    @api.response(400, "Invalid input")
    @api.response(401, "Unauthorized")
    def post(self) -> tuple[dict, int]:
        """Process a synchronous query (legacy)."""
        auth_error = _require_api_key()
        if auth_error:
            return auth_error
        request_data = request.get_json(silent=True) or {}
        user_query = request_data.get("query")
        if not user_query:
            return {"message": "Invalid input: 'query' is required"}, 400
        session_id = request_data.get("session_id")
        session_tag = request_data.get("session_tag")
        fork_from = request_data.get("fork_from")

        if fork_from:
            source_session_id = session_store.resolve_tag(fork_from) or fork_from
            session_id = session_store.fork_session(source_session_id)
        if session_tag and not session_id:
            resolved = session_store.resolve_tag(session_tag)
            session_id = resolved if resolved else None
        if not session_id:
            session_id = session_store.create_session()
        if session_tag:
            session_store.tag_session(session_id, session_tag)

        logging.info("Received user query: {}", user_query)
        task_queue: TaskQueue = orchestrate_session(
            user_query=user_query,
            session_id=session_id,
            session_store=session_store,
            approval_callback=auto_approve,
        )
        task_result = deepcopy(task_queue.task_result)
        to_return = task_queue.dict()
        to_return["task_result"] = task_result
        logging.info("Returning executed action plan.")
        to_return["session_id"] = session_id
        return to_return, 200


def main() -> None:
    """Run the Meeseeks API server."""
    app.run(debug=True, host="0.0.0.0", port=5123)


if __name__ == "__main__":
    main()

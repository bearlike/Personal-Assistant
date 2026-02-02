#!/usr/bin/env python3
"""Session transcript storage and management."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from core.common import get_logger

logging = get_logger(name="core.session_store")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SessionPaths:
    root: str
    session_id: str

    @property
    def session_dir(self) -> str:
        return os.path.join(self.root, self.session_id)

    @property
    def transcript_path(self) -> str:
        return os.path.join(self.session_dir, "transcript.jsonl")

    @property
    def summary_path(self) -> str:
        return os.path.join(self.session_dir, "summary.json")


class SessionStore:
    def __init__(self, root_dir: str | None = None) -> None:
        if root_dir is None:
            root_dir = os.getenv("MESEEKS_SESSION_DIR", "./data/sessions")
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)

    def _index_path(self) -> str:
        return os.path.join(self.root_dir, "index.json")

    def _load_index(self) -> dict:
        index_path = self._index_path()
        if not os.path.exists(index_path):
            return {"tags": {}}
        with open(index_path, encoding="utf-8") as handle:
            return json.load(handle)

    def _save_index(self, data: dict) -> None:
        with open(self._index_path(), "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        paths = self._paths(session_id)
        os.makedirs(paths.session_dir, exist_ok=True)
        return session_id

    def _paths(self, session_id: str) -> SessionPaths:
        return SessionPaths(root=self.root_dir, session_id=session_id)

    def append_event(self, session_id: str, event: dict) -> None:
        paths = self._paths(session_id)
        os.makedirs(paths.session_dir, exist_ok=True)
        payload = {"ts": _utc_now(), **event}
        with open(paths.transcript_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def load_transcript(self, session_id: str) -> list[dict]:
        paths = self._paths(session_id)
        if not os.path.exists(paths.transcript_path):
            return []
        events: list[dict] = []
        with open(paths.transcript_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning("Skipping malformed transcript line.")
        return events

    def save_summary(self, session_id: str, summary: str) -> None:
        paths = self._paths(session_id)
        os.makedirs(paths.session_dir, exist_ok=True)
        with open(paths.summary_path, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "updated_at": _utc_now()}, handle, indent=2)

    def load_summary(self, session_id: str) -> str | None:
        paths = self._paths(session_id)
        if not os.path.exists(paths.summary_path):
            return None
        with open(paths.summary_path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("summary")

    def list_sessions(self) -> list[str]:
        if not os.path.exists(self.root_dir):
            return []
        return sorted(
            name
            for name in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, name))
        )

    def fork_session(self, source_session_id: str) -> str:
        events = self.load_transcript(source_session_id)
        summary = self.load_summary(source_session_id)
        new_session_id = self.create_session()
        for event in events:
            self.append_event(new_session_id, event)
        if summary:
            self.save_summary(new_session_id, summary)
        return new_session_id

    def tag_session(self, session_id: str, tag: str) -> None:
        index = self._load_index()
        index.setdefault("tags", {})[tag] = session_id
        self._save_index(index)

    def resolve_tag(self, tag: str) -> str | None:
        index = self._load_index()
        return index.get("tags", {}).get(tag)

    def list_tags(self) -> dict[str, str]:
        index = self._load_index()
        return dict(index.get("tags", {}))

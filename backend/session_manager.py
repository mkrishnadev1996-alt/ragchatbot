import secrets
import threading
import time
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a single message in a conversation"""

    role: str  # "user" or "assistant"
    content: str  # The message content


class SessionManager:
    """Manages conversation sessions and message history"""

    def __init__(self, max_history: int = 5, ttl_seconds: float = 0, max_sessions: int = 0):
        self.max_history = max_history
        # Idle sessions older than ttl_seconds are dropped; beyond max_sessions the least
        # recently used go first. 0 disables either limit.
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._last_used: dict[str, float] = {}
        self.sessions: dict[str, list[Message]] = {}
        self.owners: dict[str, int | None] = {}
        self._lock = threading.Lock()  # requests run in a threadpool

    def create_session(self, user_id: int | None = None) -> str:
        """Create a new conversation session with an unguessable id, owned by user_id"""
        session_id = f"session_{secrets.token_urlsafe(16)}"
        with self._lock:
            self._evict(time.monotonic())
            self.sessions[session_id] = []
            self.owners[session_id] = user_id
            self._last_used[session_id] = time.monotonic()
        return session_id

    def _evict(self, now: float) -> None:
        """Drop expired sessions, then the least recently used above the cap (lock held)"""
        if self.ttl_seconds:
            for sid in [s for s, t in self._last_used.items() if now - t > self.ttl_seconds]:
                self._drop(sid)
        if self.max_sessions:
            excess = len(self.sessions) - self.max_sessions + 1  # room for the new one
            if excess > 0:
                for sid in sorted(self._last_used, key=lambda s: self._last_used[s])[:excess]:
                    self._drop(sid)

    def _drop(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        self.owners.pop(session_id, None)
        self._last_used.pop(session_id, None)

    def _alive(self, session_id: str) -> bool:
        """True if the session exists and has not idled out (refreshes its last-used time)"""
        if session_id not in self.sessions:
            return False
        now = time.monotonic()
        last = self._last_used.get(session_id, now)
        if self.ttl_seconds and now - last > self.ttl_seconds:
            self._drop(session_id)
            return False
        self._last_used[session_id] = now
        return True

    def is_owner(self, session_id: str, user_id: int) -> bool:
        """True only if the session exists and belongs to user_id"""
        with self._lock:
            return self._alive(session_id) and self.owners.get(session_id) == user_id

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the conversation history"""
        with self._lock:
            history = self.sessions.setdefault(session_id, [])
            self._last_used[session_id] = time.monotonic()
            history.append(Message(role=role, content=content))

            # Keep conversation history within limits
            if len(history) > self.max_history * 2:
                self.sessions[session_id] = history[-self.max_history * 2 :]

    def add_exchange(self, session_id: str, user_message: str, assistant_message: str):
        """Add a complete question-answer exchange"""
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", assistant_message)

    def get_conversation_history(self, session_id: str | None) -> list[dict[str, str]] | None:
        """Get conversation history as chat messages (role/content dicts) for a session"""
        if not session_id:
            return None
        with self._lock:
            if not self._alive(session_id):
                return None
            messages = list(self.sessions[session_id])  # copy: another request may append
        if not messages:
            return None

        return [{"role": msg.role, "content": msg.content} for msg in messages]

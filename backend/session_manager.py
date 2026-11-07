import secrets
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a single message in a conversation"""

    role: str  # "user" or "assistant"
    content: str  # The message content


class SessionManager:
    """Manages conversation sessions and message history"""

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.sessions: dict[str, list[Message]] = {}
        self.owners: dict[str, int | None] = {}

    def create_session(self, user_id: int | None = None) -> str:
        """Create a new conversation session with an unguessable id, owned by user_id"""
        session_id = f"session_{secrets.token_urlsafe(16)}"
        self.sessions[session_id] = []
        self.owners[session_id] = user_id
        return session_id

    def is_owner(self, session_id: str, user_id: int) -> bool:
        """True only if the session exists and belongs to user_id"""
        return session_id in self.sessions and self.owners.get(session_id) == user_id

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the conversation history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        message = Message(role=role, content=content)
        self.sessions[session_id].append(message)

        # Keep conversation history within limits
        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history * 2 :]

    def add_exchange(self, session_id: str, user_message: str, assistant_message: str):
        """Add a complete question-answer exchange"""
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", assistant_message)

    def get_conversation_history(self, session_id: str | None) -> list[dict[str, str]] | None:
        """Get conversation history as chat messages (role/content dicts) for a session"""
        if not session_id or session_id not in self.sessions:
            return None

        messages = self.sessions[session_id]
        if not messages:
            return None

        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def clear_session(self, session_id: str):
        """Clear all messages from a session"""
        if session_id in self.sessions:
            self.sessions[session_id] = []

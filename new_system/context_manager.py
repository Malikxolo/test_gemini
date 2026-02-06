"""
Context Manager for Conversation History

Maintains a rolling window of conversation history with:
- User transcripts
- Bot responses (including SILENT markers)
- Formatted output for LLM prompts
"""

import logging
from collections import deque
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConversationEntry:
    """Single conversation turn."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    was_silent: bool = False  # True if bot chose not to respond


class ConversationContext:
    """
    Manages conversation history for context-aware responses.
    
    Keeps a rolling window of the last N exchanges to provide
    the LLM with enough context for follow-up detection without
    exceeding token limits.
    """
    
    SILENT_MARKER = "[SILENT]"
    
    def __init__(self, max_entries: int = 20):
        self.max_entries = max_entries
        self.history: deque[ConversationEntry] = deque(maxlen=max_entries)
        self._last_bot_response: Optional[str] = None
        
    def add_user_message(self, transcript: str) -> None:
        if not transcript.strip():
            return
        entry = ConversationEntry(role="user", content=transcript.strip())
        self.history.append(entry)
        logger.debug(f"Added user message: {transcript[:50]}...")
        
    def add_bot_response(self, response: str, was_silent: bool = False) -> None:
        content = self.SILENT_MARKER if was_silent else response.strip()
        entry = ConversationEntry(role="assistant", content=content, was_silent=was_silent)
        self.history.append(entry)
        if not was_silent:
            self._last_bot_response = response.strip()
        logger.debug(f"Added bot response: {'SILENT' if was_silent else response[:50]}...")
        
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        messages = []
        for entry in self.history:
            messages.append({"role": entry.role, "content": entry.content})
        return messages
    
    def get_formatted_history(self) -> str:
        if not self.history:
            return "(No conversation history yet)"
        lines = []
        for i, entry in enumerate(self.history, 1):
            role_label = "User" if entry.role == "user" else "Bot"
            content = entry.content if not entry.was_silent else "[stayed silent]"
            lines.append(f"[{i}] {role_label}: {content}")
        return "\n".join(lines)
    
    def get_last_bot_response(self) -> Optional[str]:
        return self._last_bot_response
    
    def get_last_n_exchanges(self, n: int = 5) -> List[Dict[str, str]]:
        recent = list(self.history)[-n:]
        return [{"role": e.role, "content": e.content} for e in recent]
    
    def clear(self) -> None:
        self.history.clear()
        self._last_bot_response = None
        logger.info("Conversation history cleared")
        
    def __len__(self) -> int:
        return len(self.history)
    
    def __repr__(self) -> str:
        return f"ConversationContext(entries={len(self)}, max={self.max_entries})"

"""chat_memory.py

Implements a sliding-window conversational memory buffer.
"""
from collections import deque
from typing import Deque, Tuple, List


class ChatMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.buffer: Deque[Tuple[str, str]] = deque(maxlen=max_turns)

    def add_user(self, text: str):
        self.buffer.append(("User", text.strip()))

    def add_bot(self, text: str):
        self.buffer.append(("Bot", text.strip()))

    def get_context(self) -> str:
        parts: List[str] = [f"{speaker}: {text}" for speaker, text in self.buffer]
        return "\n".join(parts)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
"""Short-lived conversation memory.

An in-process store with a TTL and a session cap is right for a single node and clearly
wrong for a horizontally scaled deployment, where this becomes Redis with the same
interface. Storing turns has a data protection consequence, so entries expire and the
store never grows without bound.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Deque, List, Tuple


class ConversationMemory:
    def __init__(self, max_turns: int = 6, ttl_seconds: int = 1800, max_sessions: int = 1000) -> None:
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: "OrderedDict[str, Tuple[float, Deque[Tuple[str, str]]]]" = OrderedDict()
        self._lock = threading.Lock()

    def history(self, session_id: str) -> List[Tuple[str, str]]:
        if not session_id or self.max_turns == 0:
            return []
        with self._lock:
            self._expire()
            entry = self._sessions.get(session_id)
            if entry is None:
                return []
            self._sessions.move_to_end(session_id)
            return list(entry[1])

    def append(self, session_id: str, question: str, answer: str) -> None:
        if not session_id or self.max_turns == 0:
            return
        with self._lock:
            self._expire()
            entry = self._sessions.get(session_id)
            if entry is None:
                entry = (time.time(), deque(maxlen=self.max_turns))
            turns = entry[1]
            turns.append((question, answer))
            self._sessions[session_id] = (time.time(), turns)
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        expired = [key for key, (stamp, _) in self._sessions.items() if stamp < cutoff]
        for key in expired:
            self._sessions.pop(key, None)

    @property
    def session_count(self) -> int:
        with self._lock:
            self._expire()
            return len(self._sessions)

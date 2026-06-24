"""Minimal peer/session registry for current WebSocket fallback transport."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class PeerConnectionInfo:
    session_id: str
    transport: str = "websocket_fallback"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class PeerManager:
    def __init__(self):
        self._peers: dict[str, PeerConnectionInfo] = {}

    def upsert(self, session_id: str, transport: str = "websocket_fallback") -> PeerConnectionInfo:
        peer = self._peers.get(session_id) or PeerConnectionInfo(session_id=session_id, transport=transport)
        peer.transport = transport
        peer.updated_at = time.time()
        self._peers[session_id] = peer
        return peer

    def get(self, session_id: str) -> PeerConnectionInfo | None:
        return self._peers.get(session_id)

    def remove(self, session_id: str) -> None:
        self._peers.pop(session_id, None)


peer_manager = PeerManager()


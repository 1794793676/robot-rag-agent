"""API and gateway schemas for the realtime agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorEnvelope(BaseModel):
    ok: Literal[False] = False
    error: dict[str, Any]


class AgentSessionResponse(BaseModel):
    session_id: str
    mode: Literal["websocket_fallback", "webrtc_direct"]
    websocket_url: str
    model: str
    qwen_webrtc_allowlisted: bool = False


class WebRTCOfferRequest(BaseModel):
    session_id: str
    sdp: str
    type: Literal["offer"] = "offer"


class WebRTCAnswerResponse(BaseModel):
    session_id: str
    sdp: str = ""
    type: Literal["answer"] = "answer"
    fallback: Literal["websocket"] = "websocket"
    reason: str


class IceCandidateRequest(BaseModel):
    session_id: str
    candidate: dict[str, Any] = Field(default_factory=dict)


class AgentToolRequest(BaseModel):
    session_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentToolResponse(BaseModel):
    ok: Literal[True] = True
    session_id: str
    tool_name: str
    result: dict[str, Any]

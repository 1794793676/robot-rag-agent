"""Agent management and debug APIs."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.agent.schemas import AgentSessionResponse, AgentToolRequest, AgentToolResponse
from app.agent.session_state import session_store
from app.agent.tools import dispatch_tool_call
from app.webrtc.signaling import create_session

router = APIRouter(prefix="/api/agent", tags=["agent"])
agent_log = logging.getLogger("agent")
tool_log = logging.getLogger("tool_calls")


@router.post("/session", response_model=AgentSessionResponse)
def new_agent_session():
    payload = create_session()
    agent_log.info("session created session=%s transport=%s", payload["session_id"], payload["mode"])
    return payload


@router.post("/tool", response_model=AgentToolResponse)
async def debug_agent_tool(payload: AgentToolRequest):
    if not session_store.get(payload.session_id):
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "error": {
                    "code": "SESSION_NOT_FOUND",
                    "message": "Session not found",
                    "detail": payload.session_id,
                },
            },
        )
    tool_log.info("debug_tool_call session=%s tool=%s", payload.session_id, payload.name)
    result = await dispatch_tool_call(payload.name, payload.arguments, payload.session_id)
    return {
        "ok": True,
        "session_id": payload.session_id,
        "tool_name": payload.name,
        "result": result,
    }

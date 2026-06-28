"""WebRTC signaling API with WebSocket fallback."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

from app.agent.realtime_session import RealtimeAgentSession
from app.agent.schemas import AgentSessionResponse, IceCandidateRequest, WebRTCAnswerResponse, WebRTCOfferRequest
from app.webrtc.signaling import create_fallback_answer, create_session

router = APIRouter(prefix="/api", tags=["agent"])
webrtc_log = logging.getLogger("webrtc")


@router.post("/webrtc/session", response_model=AgentSessionResponse)
def new_webrtc_session():
    payload = create_session()
    webrtc_log.info("session created session=%s transport=%s", payload["session_id"], payload["mode"])
    return payload


@router.post("/webrtc/offer", response_model=WebRTCAnswerResponse)
def webrtc_offer(payload: WebRTCOfferRequest):
    webrtc_log.info("offer received session=%s fallback=websocket", payload.session_id)
    return create_fallback_answer(payload.session_id)


@router.post("/webrtc/ice")
def webrtc_ice(payload: IceCandidateRequest):
    webrtc_log.info("ice received session=%s fallback=websocket", payload.session_id)
    return {"ok": True, "session_id": payload.session_id, "fallback": "websocket"}


@router.websocket("/agent/ws/{session_id}")
async def agent_websocket(websocket: WebSocket, session_id: str):
    await RealtimeAgentSession(
        session_id,
        websocket,
        orchestrator=websocket.app.state.rag_first_turn_orchestrator,
    ).run()

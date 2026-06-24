"""Shared DataChannel/WebSocket message names for the browser gateway."""

CLIENT_MESSAGE_TYPES = {"interrupt", "user_text", "audio_state", "audio_chunk", "close"}
SERVER_MESSAGE_TYPES = {
    "connected",
    "text_delta",
    "audio_delta",
    "tool_call",
    "tool_result",
    "clear_audio_buffer",
    "response_cancelled",
    "response_started",
    "response_done",
    "error",
}


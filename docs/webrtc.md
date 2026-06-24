# WebRTC And Fallback

## 当前实现

Qwen-Omni-Realtime 官方支持 WebSocket 和 WebRTC。官方文档说明 WebRTC SDP 交换端点目前需要 allowlist，因此本项目优先保证可运行，采用：

```text
Browser
  -> WebSocket fallback
Backend Agent Gateway
  -> Qwen Realtime native WebSocket
```

保留的 Signaling API：

- `POST /api/webrtc/session`
- `POST /api/webrtc/offer`
- `POST /api/webrtc/ice`

`/api/webrtc/session` 返回：

```json
{
  "session_id": "sess_xxx",
  "mode": "websocket_fallback",
  "websocket_url": "/api/agent/ws/sess_xxx",
  "model": "qwen3.5-omni-flash-realtime",
  "qwen_webrtc_allowlisted": false
}
```

## 音频格式

- 浏览器输入：麦克风 Float32 PCM 重采样为 16 kHz mono int16 PCM。
- Qwen 输入：`input_audio_buffer.append`，Base64 PCM。
- Qwen 输出：`response.audio.delta`，24 kHz mono int16 PCM。
- 浏览器播放：`StreamingAudioPlayer` 将 PCM 写入 `AudioBuffer` 顺序播放。

## 直接 WebRTC 切换点

如果拿到 Qwen WebRTC allowlist endpoint，可在 `webrtc/signaling.py` 中将 offer 代理到官方 SDP endpoint，并让前端改为真正的 `RTCPeerConnection` 音频轨道传输。Function Calling 事件仍可通过 DataChannel 处理，工具执行逻辑不需要变化。


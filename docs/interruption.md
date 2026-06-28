# Interruption

打断目标是确保用户插话后旧回答立即停止，旧 response_id 的后续音频不再播放。

## 前端流程

`src/webrtc/interruptController.ts` 使用 Web Audio 音量检测：

- Agent 正在说话；
- 用户音量持续超过阈值；
- 持续时间超过 `minSpeechMs`；
- 不在 cooldown；

触发后：

1. `audioPlayer.stop()`
2. `audioPlayer.clear()`
3. 发送 `{type: "interrupt", response_id, reason}`
4. 状态切到 `interrupted`

`StreamingAudioPlayer.shouldPlay()` 会检查 response_id，不匹配的旧音频直接丢弃。

## 后端流程

`backend/app/agent/interruption.py`：

1. 校验 `session_state.current_response_id`。
2. 如果 response_id 已过期，返回 `ignored: true`。
3. 调用 `QwenRealtimeClient.cancel_response()`。
4. 设置 `interrupted = True`、`is_agent_speaking = False`、`current_response_id = None`。
5. 发送 `clear_audio_buffer`。
6. 发送 `response_cancelled`。
7. 将旧 response_id 标记为 inactive，后续音频包在后端和前端都会被丢弃。

## 测试

自动测试：

```bash
cd backend
.venv/bin/pytest tests/test_agent_core.py -q
```

真实 gateway 冒烟：

```bash
cd backend
.venv/bin/python scripts/smoke_realtime_gateway.py --interrupt
```

期望输出中：

- `sent_interrupt: true`
- `saw_clear_audio_buffer: true`
- `saw_response_cancelled: true`

手动测试：

1. 让 Agent 开始长回答。
2. 播报过程中直接说话或点击“打断”。
3. 观察音频立即停止，状态变为“已打断”。
4. 继续提问，确认新回答可以生成。

页面诊断：

1. 打开前端“实时语音 Agent”。
2. 点击“诊断 / 打断”。
3. 期望页面显示“通过”，详情包含 `clear_audio_buffer=1` 和 `response_cancelled=1`。

可直接打开：

```text
http://localhost:5173/?page=agent&diag=interrupt
```

然后在浏览器控制台检查：

```js
window.__realtimeDiagnostics.results[0]
```

## 检索阶段取消与切库

打断同样适用于 `transcribing`、`retrieving` 和 `reranking`。即使尚无 `response_id`，当前异步 turn 也会被取消，且不会进入 `response.create`。关闭连接或切库时，后端先把 session/turn 标为 `cancelled`；transcription、retrieval、rerank 完成后，以及调用 `response.create` 前，都必须通过 `session_id + connection_id + turn_id + rag_database_id` 的 current 检查。

切库顺序为：禁用输入/停止录音 → 取消旧 turn/session → 断开旧 Agent → 清空来源和音频 → 使用新 `rag_database_id` 创建 session 并自动重连。旧连接迟到事件因 `session_id`、`connection_id` 或数据库不匹配而被丢弃。

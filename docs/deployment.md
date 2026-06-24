# Deployment

目标服务器：双核 CPU、2GB 内存。推荐不使用复杂 Docker Compose，直接运行 FastAPI + 静态前端 + SQLite。

## 目录

```text
/opt/robot-rag-agent/
  backend/
  frontend/
  storage/
  logs/
  .env
```

`storage/` 保存 SQLite、上传文件和可重建索引；`logs/` 保存 Agent 和 WebRTC 日志。

## 后端

```bash
cd /opt/robot-rag-agent/backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp ../.env.example ../.env
```

编辑 `.env`：

```env
DASHSCOPE_API_KEY=your_key
QWEN_REALTIME_MODEL=qwen3.5-omni-flash-realtime
RAG_BASE_URL=http://127.0.0.1:8000
SESSION_TTL_SECONDS=1800
```

启动：

```bash
cd /opt/robot-rag-agent/backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

2GB 内存机器建议 `--workers 1`，不要开启多进程，否则内存中的 session 状态不会共享。

## systemd

`/etc/systemd/system/robot-rag-agent.service`：

```ini
[Unit]
Description=Robot RAG Realtime Agent
After=network.target

[Service]
WorkingDirectory=/opt/robot-rag-agent/backend
EnvironmentFile=/opt/robot-rag-agent/.env
ExecStart=/opt/robot-rag-agent/backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=3
User=www-data
Group=www-data

[Install]
WantedBy=multi-user.target
```

命令：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now robot-rag-agent
sudo systemctl status robot-rag-agent
sudo journalctl -u robot-rag-agent -f
```

## 前端

```bash
cd /opt/robot-rag-agent/frontend
npm ci
npm run build
```

构建产物在 `frontend/dist/`。

## nginx

示例：

```nginx
server {
    listen 80;
    server_name example.com;

    client_max_body_size 50m;

    root /opt/robot-rag-agent/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/agent/ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

浏览器麦克风在生产环境通常要求 HTTPS。上线时请配置 TLS。

## 日志

应用日志：

```bash
tail -f /opt/robot-rag-agent/logs/agent.log
tail -f /opt/robot-rag-agent/logs/webrtc.log
tail -f /opt/robot-rag-agent/logs/tool_calls.log
tail -f /opt/robot-rag-agent/logs/errors.log
```

systemd 日志：

```bash
journalctl -u robot-rag-agent -f
```

日志不记录 API Key，不落盘原始音频。

## 健康检查

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/agent/session
```

## 浏览器诊断

生产环境配置 HTTPS 后，打开：

```text
https://example.com/?page=agent&diag=stream
https://example.com/?page=agent&diag=interrupt
https://example.com/?page=agent&diag=microphone
```

`stream` 和 `interrupt` 验证浏览器页面到后端 gateway 再到 Qwen Realtime 的路径；`microphone` 验证浏览器麦克风权限。浏览器控制台可读取 `window.__realtimeDiagnostics`。

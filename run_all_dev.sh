#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
  echo "安装前端依赖..."
  (cd "$ROOT_DIR/frontend" && npm install)
fi

echo "启动后端：http://localhost:8000"
"$ROOT_DIR/backend/run_dev.sh" &
BACKEND_PID=$!

echo "等待后端健康检查通过..."
for _ in {1..90}; do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "后端进程已退出，请检查上方日志。"
    wait "$BACKEND_PID"
    exit 1
  fi
  sleep 1
done

if ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "后端健康检查超时。"
  exit 1
fi

echo "启动前端：http://localhost:5173"
(cd "$ROOT_DIR/frontend" && npm run dev) &
FRONTEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "按 Ctrl+C 停止两个开发服务。"
wait

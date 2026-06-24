#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  if ! python3 -m venv .venv; then
    echo "无法创建 .venv。Ubuntu/WSL 请先执行：sudo apt install python3-venv"
    exit 1
  fi
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if ! .venv/bin/python -c "import hnswlib" 2>/dev/null; then
  echo "尝试安装可选的 hnswlib；失败时系统自动使用 NumPy 检索。"
  .venv/bin/pip install "hnswlib>=0.8" || true
fi

exec .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "未找到 python3，请先安装 Python 3.10+。"
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"需要 Python 3.10+，当前为 {sys.version.split()[0]}")
print("Python:", sys.version.split()[0])
PY

if [ ! -d .venv ]; then
  if ! "$PYTHON_BIN" -m venv .venv; then
    echo
    echo "创建虚拟环境失败。Ubuntu/Debian 可执行："
    echo "  apt update && apt install -y python3-venv"
    exit 1
  fi
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
fi
mkdir -p logs
.venv/bin/python self_check.py

echo
echo "安装完成。"
echo "1. 编辑配置：nano $DIR/config.yaml"
echo "2. 前台测试：$DIR/run_console.sh"
echo "3. 后台启动：$DIR/start.sh"

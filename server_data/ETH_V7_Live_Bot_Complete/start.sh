#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
mkdir -p logs

if [ ! -x .venv/bin/python ]; then
  echo "尚未安装，请先执行：./install.sh"
  exit 1
fi
if [ ! -f config.yaml ]; then
  cp config.example.yaml config.yaml
  echo "已生成 config.yaml，请先填写通知或代理配置。"
fi
if [ -f v7_bot.pid ]; then
  OLD_PID="$(cat v7_bot.pid 2>/dev/null || true)"
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "V7 已在运行，PID=$OLD_PID"
    exit 0
  fi
fi

nohup env PYTHONUNBUFFERED=1 "$DIR/.venv/bin/python" "$DIR/v7_run.py" --config "$DIR/config.yaml" \
  >> "$DIR/logs/v7_console.log" 2>&1 &
PID=$!
echo "$PID" > "$DIR/v7_bot.pid"
sleep 2
if kill -0 "$PID" 2>/dev/null; then
  echo "V7 已启动，PID=$PID"
  echo "查看日志：tail -f $DIR/logs/v7_console.log"
else
  echo "启动失败，请查看：tail -n 100 $DIR/logs/v7_console.log"
  exit 1
fi

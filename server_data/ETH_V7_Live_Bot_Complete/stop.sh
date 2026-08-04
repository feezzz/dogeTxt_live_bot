#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$DIR/v7_bot.pid"
if [ ! -f "$PID_FILE" ]; then
  echo "未找到 PID 文件，机器人可能未运行。"
  exit 0
fi
PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 20); do
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
    sleep 0.5
  done
  if kill -0 "$PID" 2>/dev/null; then
    kill -9 "$PID"
  fi
  echo "V7 已停止，PID=$PID"
else
  echo "进程 $PID 已不存在。"
fi
rm -f "$PID_FILE"

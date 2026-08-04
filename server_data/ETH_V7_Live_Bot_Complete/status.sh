#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$DIR/v7_bot.pid" ]; then
  PID="$(cat "$DIR/v7_bot.pid")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "RUNNING PID=$PID"
    ps -p "$PID" -o pid,etime,cmd
    exit 0
  fi
fi
echo "STOPPED"
exit 1

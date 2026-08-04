#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
if [ ! -x .venv/bin/python ]; then
  echo "尚未安装，请先执行：./install.sh"
  exit 1
fi
exec "$DIR/.venv/bin/python" "$DIR/v7_run.py" --console --config "$DIR/config.yaml"

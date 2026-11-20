#!/usr/bin/env bash
# Starts Unmask in the background on port 8000 (CPU). Log: /tmp/unmask.log
cd "$(dirname "$0")/.."
if curl -sf http://127.0.0.1:8000/api/models >/dev/null 2>&1; then
  echo "Unmask is already running on port 8000"; exit 0
fi
setsid nohup .venv/bin/python scripts/run_app.py --device cpu --port 8000 > /tmp/unmask.log 2>&1 < /dev/null &
echo "Unmask starting on port 8000 (log: /tmp/unmask.log)"

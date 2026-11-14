#!/usr/bin/env bash
# Run the FastAPI model service and the Vite frontend.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
FRONTEND_DIR="$REPO_ROOT/frontend"

echo "=========================================================="
echo " Starting VARUNA: ANUGA and ConvLSTM flood models         "
echo "=========================================================="

BACKEND_PID=''
FRONTEND_PID=''

cleanup() {
  echo ""
  echo "Shutting down servers..."
  if [[ -n "$FRONTEND_PID" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$BACKEND_PID" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

# 1. Start Backend FastAPI Server
echo "Starting Backend API on http://localhost:8000 ..."
if [[ -x "$BACKEND_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$BACKEND_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

cd "$REPO_ROOT"
PORT=8000 "$PYTHON_BIN" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend health check..."
for i in {1..15}; do
  if curl --fail --silent http://localhost:8000/api/health >/dev/null; then
    echo "✓ Backend API is ready!"
    break
  fi
  sleep 1
done
if ! curl --fail --silent http://localhost:8000/api/health >/dev/null; then
  echo "Backend failed to start; check the Python environment and port 8000."
  exit 1
fi

# 2. Start Frontend Vite Server
echo "Starting Frontend Development Server on http://localhost:5173 ..."
cd "$FRONTEND_DIR"
npm run dev -- --host 127.0.0.1 &
FRONTEND_PID=$!

echo ""
echo "=========================================================="
echo " VARUNA System Running:"
echo " - Frontend: http://localhost:5173"
echo " - Backend API: http://localhost:8000"
echo " - API Documentation: http://localhost:8000/docs"
echo " Press Ctrl+C to stop both servers."
echo "=========================================================="

wait

#!/usr/bin/env bash
# Start backend (FastAPI) and frontend (CRA) so they actually talk to each
# other. Ctrl-C stops both.
#
# Port 8000 is avoided on purpose: VS Code plugin helpers have been
# observed binding 127.0.0.1:8000 with a SimpleHTTPServer, which silently
# shadows another process on 8000 and makes the backend look "down".
# The backend itself listens on loopback unless a launch token is set.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/canvas"
BACKEND_PYTHON="${BACKEND_PYTHON:-python}"

if ! "$BACKEND_PYTHON" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "ERROR: backend requires Python 3.10+ (the repository pins 3.12)." >&2
  echo "Set BACKEND_PYTHON to a compatible interpreter and retry." >&2
  exit 1
fi

export BACKEND_PORT="${BACKEND_PORT:-8765}"
# Loopback unless the operator set HOST. A public address is refused unless
# TENSORVIZ_LOCAL_TOKEN is set; AUTH_DISABLED is not enough.
export HOST="${HOST:-127.0.0.1}"
if ! (cd "$BACKEND_DIR" && "$BACKEND_PYTHON" -c 'import sys; from bind_policy import ensure_safe_bind; ensure_safe_bind(sys.argv[1])' "$HOST"); then
  exit 1
fi
# Keep browser requests on the frontend origin during local development. CRA's
# setupProxy forwards API traffic to the selected backend port, avoiding a
# needless CORS/private-network dependency. An explicit API URL still wins.
export REACT_APP_DEV_API_PROXY_TARGET="http://localhost:${BACKEND_PORT}"
export REACT_APP_API_URL="${REACT_APP_API_URL:-}"

# Auth must be bypassed on both sides or neither. The frontend flag lives in the
# gitignored canvas/.env.local (CRA reads it directly); the backend reads
# AUTH_DISABLED from the environment. Setting only one gives you a UI that loads
# and a backend that 401s every request, which reads as "backend didn't respond"
# in the overview — so derive the backend flag from the frontend's, and say so.
if [[ -z "${AUTH_DISABLED:-}" && -f "$FRONTEND_DIR/.env.local" ]] &&
   grep -Eq '^[[:space:]]*REACT_APP_AUTH_DISABLED[[:space:]]*=[[:space:]]*(1|true|yes|on)[[:space:]]*$' \
     "$FRONTEND_DIR/.env.local"; then
  export AUTH_DISABLED=true
  echo "Auth: DISABLED on both sides (from .env.local) — writes are attributed to local-dev-user."
fi

# Fail loud if the backend port is already in use — otherwise we'd start,
# silently lose to whatever is squatting on it, and look "disconnected".
if lsof -nP -iTCP:"${BACKEND_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "ERROR: port ${BACKEND_PORT} is already in use:" >&2
  lsof -nP -iTCP:"${BACKEND_PORT}" -sTCP:LISTEN >&2
  echo "Free it, or rerun with BACKEND_PORT=<other> ./start.sh" >&2
  exit 1
fi

backend_pid=""
cleanup() {
  if [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" 2>/dev/null; then
    echo "Stopping backend (pid $backend_pid)..."
    kill "$backend_pid" 2>/dev/null || true
    wait "$backend_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# CRA already hot-reloads the frontend. Keep the backend in lockstep during
# local development too; otherwise a pull/checkout can leave the browser using
# a newer generated contract against a stale in-memory FastAPI router (usually
# surfacing as misleading 405/422 errors). Production remains unaffected because
# this behavior belongs only to the developer launcher and can be disabled with
# BACKEND_RELOAD=false.
backend_reload="${BACKEND_RELOAD:-true}"
case "$backend_reload" in
  1|true|TRUE|yes|YES|on|ON)
    echo "Starting backend with reload: $BACKEND_PYTHON -m uvicorn server_v2:app (http://${HOST}:${BACKEND_PORT})"
    (cd "$BACKEND_DIR" && BACKEND_PORT="$BACKEND_PORT" HOST="$HOST" "$BACKEND_PYTHON" -m uvicorn \
      server_v2:app --host "$HOST" --port "$BACKEND_PORT" --reload --reload-dir "$BACKEND_DIR") &
    ;;
  0|false|FALSE|no|NO|off|OFF)
    echo "Starting backend: $BACKEND_PYTHON server_v2.py (http://${HOST}:${BACKEND_PORT})"
    (cd "$BACKEND_DIR" && BACKEND_PORT="$BACKEND_PORT" HOST="$HOST" "$BACKEND_PYTHON" server_v2.py) &
    ;;
  *)
    echo "ERROR: BACKEND_RELOAD must be true or false (received: $backend_reload)." >&2
    exit 1
    ;;
esac
backend_pid=$!

echo "Waiting for backend readiness..."
backend_ready=false
health_json=""
for _ in {1..60}; do
  if ! kill -0 "$backend_pid" 2>/dev/null; then
    echo "ERROR: backend exited before becoming ready." >&2
    wait "$backend_pid" || true
    exit 1
  fi
  if health_json="$(curl -fsS "http://localhost:${BACKEND_PORT}/health" 2>/dev/null)"; then
    backend_ready=true
    break
  fi
  sleep 1
done

if [[ "$backend_ready" != true ]]; then
  echo "ERROR: backend did not become ready within 60 seconds." >&2
  exit 1
fi

echo "Backend: ready."

frontend_api_route="${REACT_APP_API_URL:-same-origin proxy -> ${REACT_APP_DEV_API_PROXY_TARGET}}"
echo "Starting frontend: npm start (http://localhost:3000 -> ${frontend_api_route})"
cd "$FRONTEND_DIR" && npm start

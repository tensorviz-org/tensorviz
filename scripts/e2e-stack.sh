#!/usr/bin/env bash
# The browser specs, against a stack this script owns.
#
# Replaces the "browser specs against a live stack" job of the archived
# .github/workflows/frontend-gate.yml. It does what that job did, in the same order and
# with the same environment, and additionally stops what it started — a CI runner is
# thrown away afterwards and a developer's machine is not.
#
# The gate this replaced ran two entirely `describe.skip`'d specs: Playwright collected
# 8, skipped 8, exited 0, and no browser test guarded a frontend change for six weeks.
# `run-e2e.mjs` fails the run if zero tests executed, which is the only reason a green
# line here means anything.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
BACKEND_PORT="${BACKEND_PORT:-8765}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
LOGS="$ROOT/.gate/e2e"
mkdir -p "$LOGS"

export KMP_DUPLICATE_LIB_OK=TRUE
export AUTH_DISABLED=true
export BACKEND_PORT
export E2E_BACKEND_URL="http://localhost:$BACKEND_PORT"
export E2E_FRONTEND_URL="http://localhost:$FRONTEND_PORT"
export REACT_APP_API_URL="$E2E_BACKEND_URL"
export TENSORVIZ_TEST_PYTHON="$PYTHON"

backend_pid=""
frontend_pid=""

# Only ever kill what THIS script started. A developer running the specs usually has
# their own stack up on the same ports, and a cleanup that killed by port would take
# down the session they were working in.
cleanup() {
  local status=$?
  for pid in "$frontend_pid" "$backend_pid"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      # Negative pid: the whole process group. `npm start` forks react-scripts, and
      # killing only npm leaves the dev server holding port 3000 until it is found by
      # hand — which is the state that makes the NEXT run fail for the wrong reason.
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
  for pid in "$frontend_pid" "$backend_pid"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  if [ "$status" -ne 0 ]; then
    echo ""
    echo "the browser specs did not pass. Logs:"
    echo "  $LOGS/backend.log"
    echo "  $LOGS/frontend.log"
    echo "  $ROOT/canvas/playwright-report"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

# A port already in use is almost always the developer's own stack. Say so and use it
# rather than starting a second copy that will fail to bind and look like a broken gate.
listening() { curl -fsS "$1" >/dev/null 2>&1; }

if listening "http://localhost:$BACKEND_PORT/health"; then
  echo "backend: already serving on :$BACKEND_PORT — using it"
else
  echo "backend: starting on :$BACKEND_PORT"
  # Python supplies setsid on both macOS and Linux. Exec keeps the recorded pid
  # equal to the process-group leader, so cleanup also stops spawned children.
  ( cd "$ROOT/backend" && exec "$PYTHON" -c \
    'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
    "$PYTHON" server_v2.py ) > "$LOGS/backend.log" 2>&1 &
  backend_pid=$!
  for _ in $(seq 1 90); do
    listening "http://localhost:$BACKEND_PORT/health" && break
    sleep 2
  done
  if ! curl -fsS "http://localhost:$BACKEND_PORT/health" | grep -q '"status":"healthy"'; then
    echo "backend never reported healthy; see $LOGS/backend.log" >&2
    tail -20 "$LOGS/backend.log" >&2 || true
    exit 1
  fi
fi

if listening "http://localhost:$FRONTEND_PORT/"; then
  echo "frontend: already serving on :$FRONTEND_PORT — using it"
else
  echo "frontend: starting on :$FRONTEND_PORT (a cold CRA dev server takes a minute)"
  # Auth off on both sides or neither: only one gives a UI that loads against a backend
  # that 401s everything.
  ( cd "$ROOT/canvas" && BROWSER=none PORT="$FRONTEND_PORT" \
    REACT_APP_AUTH_DISABLED=true REACT_APP_API_URL='' \
    REACT_APP_DEV_API_PROXY_TARGET="$E2E_BACKEND_URL" \
    exec "$PYTHON" -c 'import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
    npm start ) > "$LOGS/frontend.log" 2>&1 &
  frontend_pid=$!
  for _ in $(seq 1 150); do
    curl -fsS "http://localhost:$FRONTEND_PORT/" 2>/dev/null | grep -q 'id="root"' && break
    sleep 2
  done
  if ! curl -fsS "http://localhost:$FRONTEND_PORT/" 2>/dev/null | grep -q 'id="root"'; then
    echo "frontend never served an app shell; see $LOGS/frontend.log" >&2
    tail -20 "$LOGS/frontend.log" >&2 || true
    exit 1
  fi
fi

# A healthy backend on its own says nothing about the page's API route. Refuse
# an externally started frontend with the wrong proxy before running any specs.
curl -fsS "$E2E_FRONTEND_URL/api/layer-types" | "$PYTHON" -c \
  'import json,sys; data=json.load(sys.stdin); assert data, "frontend API proxy returned no layer types"'

# Chromium, once. `npx playwright test` without it fails with a browser-download error
# that reads as a broken test rather than a missing install.
if ! ( cd "$ROOT/canvas" && npx playwright install chromium --dry-run >/dev/null 2>&1 ); then
  echo "playwright: installing chromium (first run only)"
  ( cd "$ROOT/canvas" && npx playwright install chromium )
fi

echo "running the substantive browser specs"
cd "$ROOT/canvas" && npm run test:e2e:substantive

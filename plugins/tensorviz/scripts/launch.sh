#!/bin/sh
# Optional Unix wrapper for the source-built Node.js host.
if ! command -v node >/dev/null 2>&1; then
  echo 'tensorViz requires Node.js 22 or newer on the coding client PATH. Install Node.js from https://nodejs.org and restart the client.' >&2
  exit 127
fi
exec node "${0%/*}/launch.cjs" "$@"

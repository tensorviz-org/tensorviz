#!/bin/sh
if ! command -v node >/dev/null 2>&1; then
  echo 'tensorViz setup needs Node.js 22 or newer. Install it from https://nodejs.org, reopen your terminal, then run this installer again.' >&2
  exit 127
fi
if ! node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 22 ? 0 : 1)'; then
  echo 'tensorViz setup needs Node.js 22 or newer. Update Node.js from https://nodejs.org, reopen your terminal, then retry.' >&2
  exit 1
fi
if [ ! -f "${0%/*}/../runtime/permissions.cjs" ]; then
  echo 'This is an incomplete tensorViz package. Extract the complete release and run its installer; a source checkout must first run npm run package --prefix plugin.' >&2
  exit 1
fi
if [ "$#" -eq 0 ]; then set -- --apply; fi
exec node "${0%/*}/install.cjs" "$@"

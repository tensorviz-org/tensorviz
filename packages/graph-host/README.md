# Shared graph host

This package owns graph opening, execution coordination, capture inspection,
review snapshots, runtime acquisition helpers and MCP graph services. The VS
Code extension and the plugin host both consume its public module exports.

It contains no VS Code imports, including type imports, and no imports from either
application or the frontend. Editor buffers, native source reviews and
permission UI remain in host adapters. Shared observations use an
optional callback; they do not send usage reports.

Run `npm ci` and `npm run build:shared` at the repository root. Consumers use
`@tensorviz/graph-host` or its module exports. Release bundlers select the
`tensorviz-source` condition; compiled output stays outside version control.

`viewTransport.ts` and `viewWire.ts` own the reusable view connection and bounded
stream decoder. Browser consumers receive checked copies through
`tooling/contracts/sync.mjs` for their independent compiler. Node injects its own
socket, location and credential handling; importing this package never opens a
connection or grants execution permission.

Layout of the repository is in the [README](../../README.md#layout).

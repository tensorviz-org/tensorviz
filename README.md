# TensorViz

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Node.js 22](https://img.shields.io/badge/node-22-339933.svg)](docs/local-setup.md#versions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](docs/local-setup.md#versions)
[![VS Code 1.106+](https://img.shields.io/badge/VS%20Code-%3E%3D1.106-007ACC.svg)](docs/local-setup.md#versions)

TensorViz shows a PyTorch or JAX/Flax model as a graph inside your editor. Open a class as a graph, read tensor shapes, focus the matching Python, expand and collapse modules, diff a change on the graph, and export a `.tensorviz.json` graph file, SVG, or PNG.

It ships as a VS Code extension (Cursor and VSCodium install the same VSIX), a coding-agent plugin for Claude Code, Codex, and Cursor, and a browser view of the same canvas. The plugin speaks MCP over stdio. The extension also serves MCP on loopback while the editor is open.

Versions, engines, and Python minima are in the [versions table](docs/local-setup.md#versions). `.python-version` is 3.12 and `.nvmrc` is 22.

`npm run release` checks a QA receipt and does not publish. Publication, the tldraw production key, and the loopback bind rule are in [local setup](docs/local-setup.md#release-checks).

## Try it

From the repository root:

```sh
npm ci && npm ci --prefix canvas && npm ci --prefix extension && npm ci --prefix plugin && npm run build:shared
npm run protocol:check
python3 -m venv venv && ./venv/bin/python -m pip install -r backend/requirements-cpu.txt pytest
npm run dev:build --prefix extension
```

Press F5 on **Run tensorViz extension (examples/tensorviz-test)**. Install steps, the VSIX, and the plugin archive are in the [quickstart](docs/quickstart.md).

## Layout

| Path | What it is |
| --- | --- |
| `extension/` | VS Code, Cursor, and VSCodium extension |
| `canvas/` | React graph UI |
| `backend/` | Python capture and projection |
| `packages/contracts`, `packages/graph-host` | Shared protocol and MCP host |
| `plugin/`, `plugins/tensorviz/` | Coding-agent plugin host and manifests |
| `examples/` | Small models |
| `docs/` | Product docs. Index: [docs/README.md](docs/README.md) |

## Docs

- [Documentation index](docs/README.md)
- [Quickstart](docs/quickstart.md)
- [Concepts](docs/concepts.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Code of conduct](CODE_OF_CONDUCT.md)

TensorViz code is [MIT](LICENSE). tldraw is not covered by that license. Read [NOTICE](NOTICE) and [docs/licensing.md](docs/licensing.md) before you give a build to other people.

The hosted website application is not in this repository. **Add to tensorViz** and **tensorViz: Sign in to tensorViz** stay in the extension. See [share a graph](docs/share-to-website.md).

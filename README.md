# TensorViz

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![VS Code 1.106+](https://img.shields.io/badge/VS%20Code-%3E%3D1.106-007ACC.svg)](docs/install.md#extension)

Explore your PyTorch or JAX/Flax model as an interactive graph. Follow tensor shapes, find the matching Python, and review changes alongside your code.

<a id="try-it"></a>

## Install

**Coding-agent plugin** (preview: Apple Silicon macOS (ARM64) and Linux x64 (glibc >=2.31), with CPython 3.12 and CPU capture):

Windows, Intel Mac, Linux ARM64, and musl are unsupported by the published standalone plugin.

Codex:

```sh
codex plugin marketplace add tensorviz-org/tensorviz-plugins
codex plugin add tensorviz@tensorviz-standalone

```

Claude Code:

```sh
claude plugin marketplace add tensorviz-org/tensorviz-plugins
claude plugin install tensorviz@tensorviz-standalone --scope user
```

Cursor: follow [Install in Cursor](https://github.com/tensorviz-org/tensorviz-plugins#install-in-cursor).

**Editor extension** for VS Code, Cursor, or VSCodium ([Marketplace](https://marketplace.visualstudio.com/items?itemName=tensorviz.tensorviz-vscode), [Open VSX](https://open-vsx.org/extension/tensorviz/tensorviz-vscode)):

```sh
code --install-extension tensorviz.tensorviz-vscode
```

All options, VSIX files, and platform limits are on the [Install](docs/install.md) page. Then open your first graph with the [quickstart](docs/quickstart.md).

Release automation now lives in this repository. The installation links above still serve the last verified packages until the [maintainer cutover](docs/releasing.md) is completed. To build from source, see [local setup](docs/local-setup.md) and [plugin build](docs/plugin-build.md). This repository is currently private, so GitHub links to it (source files, issues, releases) need read access until it is made public.

## Docs

Read the independently hosted documentation at <https://docs.tensorviz.dev/>.
To edit or preview it, follow the [docs contribution guide](CONTRIBUTING.md#documentation-contributions).
Only this repository and the docs dependencies are needed.

- [Install](docs/install.md)
- [Quickstart](docs/quickstart.md)
- [Coding-agent plugin](docs/agent-plugin.md)
- [Concepts](docs/concepts.md)
- [Documentation index](docs/README.md)

## Support and contributing

- [Plugin support and compatibility](https://github.com/tensorviz-org/tensorviz-plugins/blob/main/SUPPORT.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Code of conduct](CODE_OF_CONDUCT.md)

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

## License

TensorViz code is [MIT](LICENSE). tldraw, which renders the canvas, is not covered by that license. Read [NOTICE](NOTICE) and [docs/licensing.md](docs/licensing.md) before you give a build to other people.

The hosted website application is not in this repository. **Publish** and **tensorViz: Sign in to tensorViz** stay in the extension. See [share a graph](docs/share-to-website.md).

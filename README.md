# TensorViz

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![VS Code 1.106+](https://img.shields.io/badge/VS%20Code-%3E%3D1.106-007ACC.svg)](docs/install.md#extension)

Explore your PyTorch or JAX/Flax model as an interactive graph. Follow tensor shapes, find the matching Python, and review changes alongside your code.

TensorViz is a PyTorch model visualizer for your editor and coding agent. It opens an `nn.Module` from Python source as an interactive graph, with tensor shapes measured from one ordinary forward call. It needs no `torch.export` or ONNX conversion.

<a id="try-it"></a>

## Install

**Coding-agent plugin** (preview: macOS ARM64/x64, Linux x64 (glibc >=2.31), Linux ARM64 (glibc >=2.35), and Windows x64/ARM64, with CPython 3.12 and CPU capture):

Windows support is through native Codex entries; Windows Claude Code/Cursor and musl are outside the published acceptance scope. Runtime availability does not guarantee every client GUI, OS version, or model dependency works.

Codex:

```sh
codex plugin marketplace add tensorviz-org/tensorviz
codex plugin add tensorviz@tensorviz-standalone
```

Claude Code:

```sh
claude plugin marketplace add tensorviz-org/tensorviz
claude plugin install tensorviz@tensorviz-standalone --scope user
```

<a id="install-in-cursor"></a>

Cursor: download the **Cursor ZIP** from [this repository’s releases](https://github.com/tensorviz-org/tensorviz/releases/latest) and follow [Install in Cursor](docs/install.md#cursor).

**Editor extension** for VS Code, Cursor, or VSCodium ([Marketplace](https://marketplace.visualstudio.com/items?itemName=tensorviz.tensorviz-vscode), [Open VSX](https://open-vsx.org/extension/tensorviz/tensorviz-vscode)):

```sh
code --install-extension tensorviz.tensorviz-vscode
```

Already using the old plugin catalog? [Move your existing installation](docs/catalog-migration.md) before updating. All options, VSIX files, and platform limits are on the [Install](docs/install.md) page. Then open your first graph with the [quickstart](docs/quickstart.md).

Release automation and the installation catalog live in this public repository. The links above serve published packages; historical 0.x downloads retain their original locations. The source manifests declare the 1.0.0 release candidate; the installed version changes after publication. See [release operations](docs/releasing.md), [local setup](docs/local-setup.md), and [plugin build](docs/plugin-build.md).

## Docs

Read the independently hosted documentation at <https://docs.tensorviz.dev/>.
To edit or preview it, follow the [docs contribution guide](CONTRIBUTING.md#documentation-contributions).
Only this repository and the docs dependencies are needed.

- [Install](docs/install.md)
- [Quickstart](docs/quickstart.md)
- [Coding-agent plugin](docs/agent-plugin.md)
- [Concepts](docs/concepts.md)
- [Documentation index](docs/README.md)

## Guides

Comparisons and how-tos on [tensorviz.dev](https://tensorviz.dev/):

- [How to visualize a PyTorch model architecture](https://tensorviz.dev/guides/visualize-pytorch-model-architecture/): `print(model)`, torchinfo, torchview, TorchVista, TensorBoard, Netron, Google Model Explorer and TensorViz on one small CNN.
- [PyTorch model visualization tools compared](https://tensorviz.dev/guides/pytorch-model-visualization-tools/): TensorViz, Netron, torchview, Model Explorer, TensorBoard, torchinfo and torchviz.
- [Netron alternative for PyTorch](https://tensorviz.dev/guides/netron-alternative-for-pytorch/): when to keep Netron for saved files and when to use a forward-pass tool.
- [See what changed in a PyTorch model between two commits](https://tensorviz.dev/guides/diff-pytorch-model-between-git-commits/): architecture, checkpoint and metric changes.

## Support and contributing

- [Plugin support and compatibility](https://github.com/tensorviz-org/tensorviz/blob/main/SUPPORT.md)
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


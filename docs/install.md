# Install

![The shared Canvas: explore model layers after opening a model in TensorViz](images/explore-graph.png)

Choose one way to use TensorViz:

- [Coding-agent plugin](#plugin) for Codex, Claude Code, or Cursor.
- [Editor extension](#extension) for VS Code, Cursor, or VSCodium.
- [Build from source](#source) to work on this repository.

Use one TensorViz host per agent. Do not connect the same agent to both the plugin and the extension.

<a id="plugin"></a>

## Coding-agent plugin

You need your client’s terminal CLI: [Codex CLI](https://developers.openai.com/codex/cli/) or [Claude Code](https://code.claude.com/docs/en/setup).

The published plugin is a **preview** for Apple Silicon macOS with CPython 3.12 and CPU capture. It downloads a verified compiled runtime on first connection. You need internet access for that download, but no separate Node.js installation.

It is distributed from the public catalog [tensorviz-org/tensorviz-plugins](https://github.com/tensorviz-org/tensorviz-plugins). The catalog name is `tensorviz-standalone`.

### Codex

```sh
codex plugin marketplace add tensorviz-org/tensorviz-plugins
codex plugin add tensorviz@tensorviz-standalone
```

### Claude Code

```sh
claude plugin marketplace add tensorviz-org/tensorviz-plugins
claude plugin install tensorviz@tensorviz-standalone --scope user
```

### Cursor

Follow [Install in Cursor](https://github.com/tensorviz-org/tensorviz-plugins#install-in-cursor). The current Cursor preview package is [tensorviz-cursor-0.3.61.zip](https://github.com/tensorviz-org/tensorviz-plugins/releases/download/v0.3.61/tensorviz-cursor-0.3.61.zip).

### T3 Code

T3 Code uses your Codex profile. Install with the Codex commands above, in the same profile T3 uses. Details are in the [public catalog](https://github.com/tensorviz-org/tensorviz-plugins).

After installing, start a new chat. An existing chat keeps the tools it already loaded. Next: [coding-agent plugin](agent-plugin.md).

<a id="extension"></a>

## Editor extension

The published extension ID is `tensorviz.tensorviz-vscode`.

- [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=tensorviz.tensorviz-vscode)
- [Open VSX](https://open-vsx.org/extension/tensorviz/tensorviz-vscode)

From a terminal, use the command for your editor:

::: code-group

```sh [VS Code]
code --install-extension tensorviz.tensorviz-vscode
```

```sh [Cursor]
cursor --install-extension tensorviz.tensorviz-vscode
```

```sh [VSCodium]
codium --install-extension tensorviz.tensorviz-vscode
```

:::

To install a downloaded file instead, download the [0.1.36 VSIX](https://open-vsx.org/api/tensorviz/tensorviz-vscode/0.1.36/file/tensorviz.tensorviz-vscode-0.1.36.vsix), then run:

```sh
code --install-extension ./tensorviz.tensorviz-vscode-0.1.36.vsix
```

Or run **Extensions: Install from VSIX…** in the editor and choose the file.

The extension needs VS Code 1.106 or newer. Your model's Python must be 3.10 or newer. TensorViz runs its own service with a separate Python 3.11 or newer.

Next: [quickstart](quickstart.md).

<a id="source"></a>

## Build from source

Build this repository to develop TensorViz or use your own source-built version.

- [Local setup](local-setup.md): versions, dependencies, and running the extension from source.
- [Plugin build](plugin-build.md): building a plugin archive from source.

Follow the build guide before installing a source package.

## Versions and identities

Published packages are released separately and may differ from this source checkout.

| | Published | This source tree |
| --- | --- | --- |
| Extension ID | `tensorviz.tensorviz-vscode` | `tensorviz.tensorviz` |
| Plugin ID | `tensorviz@tensorviz-standalone` | Same ID, built locally |
| Versions | Shown in the marketplace or catalog | [Versions table](local-setup.md#versions) |

A source build and a published install use different extension IDs, so installing both gives you two copies. Keep one.

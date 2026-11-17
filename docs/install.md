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

The compiled plugin is a **preview** for macOS ARM64/x64, Linux x64 (glibc >=2.31), Linux ARM64 (glibc >=2.35), and Windows x64/ARM64, with CPython 3.12 and CPU capture. It downloads a verified compiled runtime on first connection. You need internet access for that download, but no separate Node.js installation.

Windows support is through native Codex entries; Windows Claude Code/Cursor and musl are outside the published acceptance scope. Runtime availability does not guarantee every client GUI, OS version, or model dependency works. See [Linux SSH runtime](linux-ssh-runtime.md) for execution-host requirements and the limits of local validation. Published platform availability does not establish real SSH-network or installed-client acceptance.

Release publication requires native CI verification of the six release targets
and release migration checks. This is
not a claim that every desktop GUI, older OS, or SSH setup has been tested.
TensorViz startup/service compatibility is separate from PyTorch wheel and model
compatibility: a working runtime does not imply a compatible PyTorch installation.
For the exact published assets and limitations, see the
[support statement](https://github.com/tensorviz-org/tensorviz/blob/main/SUPPORT.md).

It is distributed from the public catalog [tensorviz-org/tensorviz](https://github.com/tensorviz-org/tensorviz). The catalog name is `tensorviz-standalone`.

Already installed from the old repository? Follow
[Move an existing plugin to the public catalog](catalog-migration.md) first.

### Codex

```sh
codex plugin marketplace add tensorviz-org/tensorviz
codex plugin add tensorviz@tensorviz-standalone
```

### Claude Code

```sh
claude plugin marketplace add tensorviz-org/tensorviz
claude plugin install tensorviz@tensorviz-standalone --scope user
```

### Cursor

Download the **Cursor ZIP** from [this repository’s releases](https://github.com/tensorviz-org/tensorviz/releases/latest). For a fresh installation, extract it into `~/.cursor/plugins/local/tensorviz` so that folder directly contains `plugin.json`, `.cursor-plugin/plugin.json`, `scripts/`, and `skills/`. Restart Cursor and check **Customize** for the TensorViz skill and MCP server. Keep the previous installation separately for rollback; do not merge files from different versions. Local plugin imports must be allowed by your account or organization.

### T3 Code

T3 Code uses your Codex profile. Install with the Codex commands above, in the same profile T3 uses. Details are in the [public catalog](https://github.com/tensorviz-org/tensorviz).

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

To install a downloaded file instead, download the current VSIX from [Open VSX](https://open-vsx.org/extension/tensorviz/tensorviz-vscode), then run:

```sh
code --install-extension ./tensorviz.tensorviz-vscode-VERSION.vsix
```

Replace `VERSION` with the downloaded package version. Or run **Extensions: Install from VSIX…** in the editor and choose the file.

The editor extension supports macOS, Windows, and Linux with compatible Python and model dependencies; the standalone plugin's platform restrictions above do not define extension support. The extension needs VS Code 1.106 or newer. Your model's Python must be 3.10 or newer. TensorViz runs its own service with a separate Python 3.11 or newer.

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
| Extension ID | `tensorviz.tensorviz-vscode` | Same ID: `tensorviz.tensorviz-vscode` |
| Plugin ID | `tensorviz@tensorviz-standalone` | Same ID, built locally |
| Versions | Shown in the marketplace or catalog | [Versions table](local-setup.md#versions) |

A source-built VSIX uses the same extension ID as the published package, so installing it replaces that package in the same editor profile. Use a separate editor profile to keep published and development builds apart.

The source manifests declare the 1.0.0 release candidate. The catalog may serve
verified 0.1.48 packages until 1.0.0 publication completes; check the version in
the install result. Historical 0.x downloads keep their original locations.


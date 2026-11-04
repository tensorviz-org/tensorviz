# tensorViz coding-agent plugin — reference

<a id="tensorviz-coding-agent-plugin"></a>

For the short walkthrough, see [the user guide](../agent-plugin.md). This page covers locally built source-development archives. Published plugin installation is in the user guide.

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Permissions](#permissions)
- [Model graph panel](#model-graph-panel)
- [What the plugin runs](#what-the-plugin-runs)
- [Build](#build)

The plugin lets Claude Code, Codex, or Cursor open a local PyTorch model, capture
one forward pass, and read the graph. It is the same product as the VS Code
extension, in a different host. Current versions: extension **0.1.36**, plugin
host **0.3.61**.

| Package | Source | What you install |
| --- | --- | --- |
| Coding-agent plugin | `plugin/` and `plugins/tensorviz/` | A local archive built from this repo, or the plugin directory after that build |
| VS Code extension | `extension/` | A VSIX. See the [quickstart](../quickstart.md) |

Both display as **tensorViz**. The plugin id used by the Codex installer is
`tensorviz@tensorviz-standalone`. The extension id is `tensorviz.tensorviz`.
Those ids are stable. Renaming one would install a second copy.

Use one TensorViz host per agent. The plugin has its own stdio MCP server. The
extension has a separate loopback HTTP endpoint, documented in [MCP](../mcp.md).
Do not point the same agent at both.

The VS Code extension is optional. It is not required to run the plugin, and the
plugin does not install the extension.

## Requirements

- Node.js on the agent's `PATH`. The version is in the [versions table](../local-setup.md#versions).
- A Python that meets the tool-service minimum in that same table. Capture uses a project environment you
  select, with that model's dependencies.
- A local model project. tensorViz does not ask for its own provider API key.
  Your coding client's account still applies.

Build the host before you install it. This repository does not publish a
compiled runtime or a marketplace release. See [Plugin build](../plugin-build.md).

## Install

For Codex, use the built `tensorviz-<version>.tar.gz` source-development archive:

1. Check the archive against its `.sha256` file.
2. Extract it into a directory you will keep. Keep the previous extract if you
   want a rollback copy.
3. From your model checkout, run the archive's installer:

```sh
sh /path/to/tensorviz/plugins/tensorviz/scripts/install.sh
```

The installer needs the Node.js version in the [versions table](../local-setup.md#versions). It refuses to run against a bare git checkout
that has not been packaged, because the packaged host is not in `plugins/tensorviz/`
until you build. A preflight without changes:

```sh
node /path/to/tensorviz/plugins/tensorviz/scripts/install.cjs
```

Installation is for that agent's user profile on this machine, not inside the
model repo. Start a new chat afterward. An existing chat keeps the tools it
already loaded.

The guided installer configures Codex only. For Claude Code, register the extracted archive root and install the plugin explicitly:

```sh
claude plugin marketplace add /absolute/path/to/extracted-archive
claude plugin install tensorviz@tensorviz-standalone --scope user
```

The plugin folder alone is not the marketplace catalog. The Cursor archive has a different layout and cannot use the Codex guided installer. Cursor can instead use the source-built editor VSIX and **Connect coding agent (MCP)**; see [plugin build](../plugin-build.md).

If an older TensorViz connection is already enabled, choose **Use the tensorViz
plugin** in the guided installer. It disables matching user-owned connections
and keeps a recovery receipt. A manual entry named exactly `tensorviz` is moved
to a disabled name such as `tensorviz-previous`, because Codex can hide plugin
tools while that exact name exists. Unrelated connections are left alone.
Project-owned MCP settings are reported, not overwritten.

## Permissions

The installer asks once. Accepting covers local project reads, Python
inspection, model and setup execution, saving `tensorviz.toml`, and installing
reviewed dependencies into the selected environment. It applies to this OS user,
across projects. Model code runs with your user permissions. Graph text returned
to the agent can become part of that client's conversation.

The Codex setting the installer writes is only:

```toml
[plugins."tensorviz@tensorviz-standalone".mcp_servers.tensorviz]
default_tools_approval_mode = "approve"
```

Other plugins, global approval settings, and per-tool overrides stay as they
were. Organization policy still wins. The installer reports a policy it cannot
change.

Status and revocation, from the installed plugin:

```sh
node /path/to/tensorviz/plugins/tensorviz/scripts/permissions.cjs --status
node /path/to/tensorviz/plugins/tensorviz/scripts/permissions.cjs --revoke
```

Revoke stops later execution and setup. It does not kill a worker that is
already running. Read-only views of a graph you already captured stay available.
A declined decision is remembered, so the plugin stops instead of asking on
every call. `open_graph` can return `action: request_permission`; that asks
again, and it cannot grant permission by itself.

The consent file is `permissions/standalone.json` under the user data directory
from `applicationData()` in `plugin/src/environment.ts`:

| Platform | Directory |
| --- | --- |
| macOS | `~/Library/Application Support/TensorViz/standalone` |
| Windows | `%LOCALAPPDATA%\TensorViz\standalone` |
| Linux | `$XDG_DATA_HOME/tensorviz/standalone`, or `~/.local/share/tensorviz/standalone` when `XDG_DATA_HOME` is unset |

It is separate from the project and from the plugin install. Uninstalling the
plugin does not delete the consent file. Revoke it explicitly.

Unknown inputs and interpreter choices still need an answer when the project
does not resolve them. Consent does not start a model on its own, and it does
not replace an installed Torch or CUDA build.

## Model graph panel

After the plugin is connected, ask the agent to open the **Model graph** panel
with `open_graph_panel`, or use the client's plugin view if it has one. An empty
panel does not start Python. Ask the agent to `open_graph` on the model you
want. The panel shows the same canvas as the plugin browser: shapes, inputs,
refresh, and local graph review.

`open_graph_panel` returning does not prove the panel drew the graph. A current
view from `list_models`, after the canvas reports a render, is what shows that
the panel is up. Clients without that panel use `open_graph_view` and a local
browser page.

Closing the panel does not cancel a capture or edit source. Reopen **Model
graph** if the panel disconnects; the capture is kept. Up to eight panels can
attach to one plugin process. A panel with no traffic expires after five
minutes.

The panel and the browser talk to the local Node host. They do not receive the
loopback credential, and they cannot grant execution permission.

## What the plugin runs

The plugin service is `backend/standalone_server.py`. It binds **127.0.0.1**
on a free port, requires a launch token, and does not bind `0.0.0.0`. The
browser only accepts that loopback host. The shared backend
(`start.sh`, `backend/server_v2.py`) also defaults to `127.0.0.1` and refuses a
non-loopback bind unless `TENSORVIZ_LOCAL_TOKEN` is set. `AUTH_DISABLED` does
not allow that bind. See [service bind](../local-setup.md#service-bind).

Capture uses the project Python you select. The tool service can use a separate
interpreter. Changing the capture environment does not restart the service.
See [environment recovery](environment-recovery.md).

Saved `tensorviz.toml` recipes are reused. Missing setup is asked for, or the
agent can fill it through `open_graph`. **Inspect source only** does not execute
the model. After you save a source edit, refresh the graph to capture again.

## Build

```sh
npm run build --prefix plugin
npm run build:browser --prefix plugin
npm run build:panel --prefix plugin
npm run package:source-development --prefix plugin
```

`npm run package:source-development --prefix plugin` writes
`plugin/dist/tensorviz-<version>.tar.gz` and
`plugin/dist/tensorviz-cursor-<version>.tar.gz`, each with a `.sha256` file.
Details, including the tldraw key and what this repo does not publish, are in
[Plugin build](../plugin-build.md).

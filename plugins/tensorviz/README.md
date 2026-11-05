# tensorViz plugin

Explore a local PyTorch model from Claude Code, Codex, or Cursor. The plugin
captures one forward pass with the example inputs you save, shows tensor shapes,
and lets the agent read that graph. The VS Code extension is a separate install.

## Requirements

- Source development uses Node.js and Python from the [versions table](../../docs/local-setup.md#versions). The compiled preview bundles Node.js.
- The compiled preview requires CPython 3.12 on macOS Apple Silicon with compatible PyTorch and model dependencies. Its native modules require that exact Python ABI; the extension’s broader Python support does not apply to this build.
- A local PyTorch model project. No TensorViz account or provider API key is required; your coding client’s own account and permissions still apply.

This directory is the plugin manifest and installer scripts. It is not a
compiled runtime. `scripts/install.sh` refuses to run until the host has been
packaged. Build first: [docs/plugin-build.md](../../docs/plugin-build.md).

## Install

1. Build the source-development archive (`npm run package:source-development --prefix plugin`). It writes `plugin/dist/tensorviz-<version>.tar.gz` and `plugin/dist/tensorviz-cursor-<version>.tar.gz`, each with a `.sha256` file.
2. Extract the archive you need and check the `.sha256` file.
3. From your model checkout, run `node "/path/to/release/plugins/tensorviz/scripts/install.cjs"`.

Source-built managed runtimes are pinned for macOS arm64, Linux x64, and
Windows x64. Intel macOS and Linux arm64 capture are unsupported. MCP launches
through Node.js 22+ directly; Windows does not need Unix `sh`. Live Windows and
macOS runs remain untested for these platform changes.

That installs the plugin for the agent user profile. Keep one tensorViz host
enabled per agent. Start a new chat and ask the agent to open the model.

Cursor can install the VS Code extension from this repository instead. See
[docs/agent-plugin.md](../../docs/agent-plugin.md).

## Permissions

The installer asks once before model execution and dependency setup. Python
runs as your user. Revoke permission with the installed
`scripts/permissions.cjs --revoke`. Details are in the
[plugin guide](../../docs/agent-plugin.md#permissions).

## License

TensorViz code is MIT. A built plugin writes `DEPENDENCY_NOTICES.md` and
`licenses/` next to the host. The tldraw SDK keeps its own license. See
[docs/licensing.md](../../docs/licensing.md).

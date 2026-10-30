# Quickstart

Install TensorViz, then open your first PyTorch model. For JAX and Flax in the editor, see the [framework guide](jax-support.md).

<a id="install-the-extension"></a>
<a id="plugin-instead-of-the-extension"></a>

## Install

<!-- install-options:start -->
**Coding-agent plugin** (preview: Apple Silicon macOS, CPython 3.12, CPU capture)

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

**Editor extension** for VS Code 1.106+, Cursor, or VSCodium:

```sh
code --install-extension tensorviz.tensorviz-vscode
```

Use `cursor` or `codium` in place of `code` for those editors. Or install from the [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=tensorviz.tensorviz-vscode) or [Open VSX](https://open-vsx.org/extension/tensorviz/tensorviz-vscode).

**Build from source:** [local setup](local-setup.md).
<!-- install-options:end -->

All options, VSIX files, and platform limits: [Install](install.md).

<a id="open-the-example"></a>

## Open your first graph

1. **Open a project.** Use your model’s folder. Need an example? In the extension, run **tensorViz: New model from template**.
2. **Open the graph.** In the editor, click **Open as Graph** above your model. In your agent, start a new chat and ask: “Use TensorViz to show my model’s graph.”
3. **Confirm setup.** Choose Python and example inputs if asked (plugin preview: CPython 3.12). TensorViz asks before library changes and saves your setup for next time.

Expand a module, inspect its shapes, and select a layer to follow the source.

> **Capture runs your code.** Opening a graph executes one forward call of the model with the inputs you chose. **Inspect source only** opens a preview without running the model.

<a id="a-ten-line-model"></a>

Need example code? [Try the small model](getting-started.html#example).

<details id="if-opening-stops">
<summary>If opening stops</summary>

| What you see | What to do |
| --- | --- |
| No Python, or import errors | [Environment recovery](environment-recovery.md) and [model libraries](graph-libraries.md) |
| **Choose a setup method** | [Setup and inputs](graph-setup.md) |
| Capture failed | Fix the source or the inputs, then **Run again** |
| Agent cannot see tools | Start a new chat. For the extension, see [Connect MCP](mcp.md) |

</details>

## Next steps

- [Explore the graph](canvas.md)
- [Follow the code](code-focus.md)
- [Review model changes](graph-diff-review/README.md)

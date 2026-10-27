# Quickstart

Open one model as a graph. Engines and Python minima are in the [versions table](local-setup.md#versions). Build commands in full are in [local setup](local-setup.md).

## Install the extension

1. From the repository root, install and build as in [Try it](../README.md#try-it), including `npm run protocol:check` and `npm run dev:build --prefix extension`.
2. Build a VSIX:

```sh
npm run build:webview --prefix extension
npm run vsix --prefix extension
```

`npm run vsix --prefix extension` builds the webview itself. The file lands under `extension/`.

3. In VS Code, Cursor, or VSCodium, run **Extensions: Install from VSIX…** and choose that file.
4. Open a folder and choose **tensorViz** in the activity bar.

The extension id is `tensorviz.tensorviz`. Marketplace publishing is not part of this repository.

For day-to-day development, skip the VSIX: press F5 on **Run tensorViz extension (examples/tensorviz-test)**.

## Open the example

1. Open `examples/tensorviz-test`.
2. Run **Open as Graph** on `TinyTransformer` or `Simple` (CodeLens above the class, the Models view, or **tensorViz: Open as Graph**).
3. If TensorViz asks for a Python, point it at an environment that can import the model. It asks before installing libraries.
4. The saved profiles `transformer` and `simple` live in that folder's `tensorviz.toml`. A complete profile captures one forward call without a Run click.

You should see the class as a graph in the sidebar Canvas, with shapes for that saved call. **Inspect source only** opens a preview and does not run the model.

`train.py` in that example needs the local SDK first. From `examples/tensorviz-test`:

```sh
pip install -e sdk
python train.py
```

`sdk/` provides `tensorviz.experiment`, `tensorviz.node`, and `tensorviz.edge`. It records labels. It is not the TensorViz extension.

## If opening stops

| What you see | What to do |
| --- | --- |
| No Python, or import errors | [Environment recovery](environment-recovery.md) and [model libraries](graph-libraries.md) |
| **Choose a setup method** | [Setup and inputs](graph-setup.md). **Set up with agent** in VS Code, **Copy setup request** in the plugin browser, or **Set up manually** |
| Capture failed | The execution strip keeps the failure. Fix the source or the profile, then **Run again** |
| Agent cannot see tools | [Connect MCP](mcp.md). Keep the editor open for the extension endpoint |

## Starter models

With a folder open and no discovered models, the Canvas starting screen offers **Try a starter model**. That runs **tensorViz: New model from template**. **MLP classifier** and **Small CNN** are available before the backend starts.

1. Open an empty folder and open **tensorViz** in the activity bar.
2. Choose **Try a starter model**, pick **MLP classifier** or **Small CNN**, and name the file. The file is written under `models/`. An existing file is not overwritten. Cancelling writes nothing.
3. Run **Open as Graph** on the generated class.

Opening the starting screen does not install packages or execute a model. Choosing a starter does not open the graph by itself.

Without a folder, the screen offers **Open Folder**. When models are already listed, use those. Templates stay in the Command Palette and the Templates view.

## A ten-line model

The sample below is also in [getting-started.html](getting-started.html#example), which the backend test executes as written. `# tensorviz: input=2,4` belongs to the class it precedes (or to the next top-level class). It is the example input when no `tensorviz.toml` profile and no single in-file example call supply one. The three source directives are in the [reference](reference.md#source-directives).

```python
import torch.nn as nn

# tensorviz: input=2,4
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.output = nn.Linear(8, 2)

    def forward(self, x):
        return self.output(self.relu(self.hidden(x)))
```

**Open as Graph** on `Net`, then **Change inputs** if you want a different shape. The capture covers the input you save.

## Plugin instead of the extension

The directory `plugins/tensorviz` is a manifest. It is not an installable plugin until you build an archive. [Plugin build](plugin-build.md) writes `plugin/dist/tensorviz-<version>.tar.gz` and `plugin/dist/tensorviz-cursor-<version>.tar.gz`. Extract one and run its `plugins/tensorviz/scripts/install.sh`. Steps: [coding-agent plugin](agent-plugin.md#install).

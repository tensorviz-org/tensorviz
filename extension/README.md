# tensorViz — PyTorch & JAX Model Visualizer

## Contents

- [Your first graph](#your-first-graph)
- [JAX and Flax](#jax-and-flax)
- [Explore and edit](#explore-and-edit)
- [Share a graph file](#share-a-graph-file)
- [Capture and settings](#capture-and-settings)
- [Support](#support)

**Follow tensor shapes through your model, beside the Python you are editing.**

tensorViz opens a PyTorch `nn.Module` as an interactive graph in VS Code.
Expand nested modules, inspect shapes from a captured forward pass, and jump
from a layer to its source. Plain JAX functions and Flax modules also open as
traced programs with inferred shapes.

Shapes describe the example input you capture.

- **Understand a model:** explore one block at a time instead of reading the whole definition.
- **Investigate a shape mismatch:** follow the shapes recorded for your example inputs.
- **Review architecture changes:** compare a Python diff or pull request beside the graph.

PyTorch captured shapes describe one input and execution path. Graph diff compares source
without executing either revision; unresolved coverage stays visible.

Free to install. Core graph exploration and source forms work without a coding agent.

[Get started](https://tensorviz.dev/install.html#first-graph) ·
[Watch the walkthrough](https://tensorviz.dev/install.html#walkthrough) ·
[Try the Model Explorer](https://tensorviz.dev/explore/) ·
[Examples and issues](https://github.com/tensorviz-org/tensorviz)

Source, examples, and issues are in this repository:
[tensorviz-org/tensorviz](https://github.com/tensorviz-org/tensorviz).

The browser examples are recorded snapshots and need no installation. Open your
own model in VS Code to capture it in your selected Python environment.

## Your first graph

1. **Install and choose a model.** Install tensorViz in VS Code 1.106 or newer
   and open your model's folder. To start with an example, run **tensorViz: New
   model from template** and choose **MLP classifier** or **Small CNN**. Both
   starters are included in the extension.
2. **Open the graph and prepare Python.** Click **Open as Graph** above the
   `nn.Module` class, or choose it in tensorViz's Models view. Reuse your compatible
   project Python or choose a Conda/venv when asked. tensorViz checks dependencies
   and asks before downloads or library changes; existing Torch/CUDA versions are
   preserved. Initial setup can take time.
3. **Capture and explore.** tensorViz reuses saved setup or prepares example
   inputs. Fill in any missing constructor values or inputs beside the source-only
   preview. Completing setup saves `tensorviz.toml` and captures one forward call.
   Once shapes are measured, expand a block, select a layer and follow its source
   link. After editing the model, save and choose **Refresh graph** to capture again.

Capture and service Python minima are in the [versions table](../docs/local-setup.md#versions).
The capture environment needs compatible PyTorch and your model's dependencies.
The service can use a managed environment after your approval. **Inspect source only** does not execute your model. No
repository checkout, Node.js or coding agent is required.

[Full setup and troubleshooting](https://tensorviz.dev/install.html#first-graph) ·
[Remote SSH](https://tensorviz.dev/install.html#remote) ·
[Python and platform compatibility](https://tensorviz.dev/install.html#compatibility)

Use **Show Canvas / Hide Canvas** or **Ctrl+Cmd+G** on macOS
(**Ctrl+Alt+G** on Windows/Linux) to toggle the sidebar without losing your selected
model. **Open Canvas in Browser** opens it in a separate browser view. If setup
fails, its inline controls let you retry; **tensorViz: Select Python Environment**
changes the model environment and **tensorViz: Repair runtime** repairs only the
separate tensorViz runtime. See the [opening guide](https://tensorviz.dev/install.html#first-graph).

## JAX and Flax

Install compatible `jax` and, for Flax models, `flax` in your selected Python.
Choose **Open as Graph** above a JAX function or Flax Linen/NNX class and enter
explicit example input shapes. For example:

```python
import jax.numpy as jnp

def net(x):
    return jnp.tanh(x @ jnp.ones((3, 4)))
```

An input of shape `2, 3` produces a traced graph with output shape `2, 4`.
The saved `tensorviz.toml` profile has `framework = "jax"`. Linen modules use
the saved seed to initialize fresh parameters; use a Python setup factory to
bind trained variables or construct NNX state.

JAX support traces on the saved CPU or CUDA device and labels shapes as **inferred**. Control
flow primitives retain their boundary and report partial coverage; tracing does
not claim a branch executed. PyTorch source edits are unavailable for these
graphs. The shared capture worker still requires Torch.

## Explore and edit

Pan, zoom, search, expand nested modules and use the Overview minimap. Select a
node and click its file:line link at inspection zoom to open its source. Moving the editor cursor
into a mapped layer focuses it on the graph. Parse errors leave the last graph
visibly stale until the source is fixed.

Turn on **Code focus** in the Canvas header to open each selected node's source
beside the graph; double-click or Enter on a layer moves keyboard focus to it. The
chevron beside it can dim unrelated code. With the Canvas maximized, showing source
restores the split so the code appears beside it.

To use your existing coding agent, run **tensorViz: Connect coding agent (MCP)**.
A status-bar item shows Off, Ready or Connected. The connection panel can configure
a supported agent after approval or provide the URL for a Streamable HTTP client.
The agent's `open_graph` tool opens and captures a model, completes missing setup
and requests dependency installation in TensorViz's reported capture environment.
It also inspects graphs, focuses a view and proposes source edits. Each opens in
VS Code's native diff: choose **Apply source change** or **Discard proposal** in its
toolbar. Accept/Apply makes one undoable edit, saves the reviewed files and runs
the model with its saved setup; native Undo restores the source edit. Changed source
or dependencies invalidate an old proposal. Canvas chat can propose changes across
existing graph-recorded Python files. Native and optional ACP connections reuse
conversation context and attach [compact model facts](../docs/model-context.md);
preparing chat can use the agent’s provider before Send.
The MCP connection uses your existing coding agent. See
[MCP setup](https://tensorviz.dev/install.html#agents).

Edits stay within supported constructs in one owning class. Shared definitions
affect every instance. Grouping changes parameter paths and checkpoint keys.
Source review checks syntax and graph consistency, not runtime tensor correctness.

## Share a graph file

In **Open Canvas in Browser**, choose **Export graph** to save a `.tensorviz.json` snapshot. The editor Canvas offers **Publish** to share through your website account.
**Open graph file viewer** opens the independent viewer for pan, zoom, search,
module expansion, saved views, walkthroughs and layer/connection discussions.
Choose an opening view for recipients, or compare two files to review architecture
changes. Download the annotated file to share it. Recipients need a static viewer page and the
file; they do not need Python or the extension. Open the
[public graph viewer](https://tensorviz.dev/viewer/index.html) to explore a snapshot.

## Capture and settings

After changing source, save and choose **Refresh graph** to capture again.
**Change inputs** edits the saved setup. Generated dimensions are example values,
not recovered training settings; a successful capture covers that invocation only.
Choose **Model creation → Python function** in model setup to reuse an existing model builder or
return the selected component from a pipeline. Modules can remain in their original
files. The factory must return the selected class. See
[input setup guide](https://tensorviz.dev/install.html#first-graph).

Saving changed model code or a tracked Python dependency automatically validates
the open model by default, using its saved inputs, device and environment. Set
`tensorviz.traceOnSave` to `false` to run manually. Typing refreshes the source
projection without running; unchanged saves do not run. Missing or ambiguous setup refuses execution.

Execution caching is on by default for eligible identical CPU runs. A brief Python
safety check still runs; the model does not run again. Use **Run fresh** to execute
it again, or set `tensorviz.executionCache.mode` to `off`.

`tensorviz.traceTier` defaults to `real`. Experimental `fake` uses shape-only
tensors; `auto` uses them after a real run counts at least 150 million parameters.
Models needing tensor values retry with real tensors. Saved profiles, validation
and candidates always use real tensors.

All 15 settings, including `tensorviz.traceTier`, `tensorviz.autoOpenCanvas`, `tensorviz.followEditor`, `tensorviz.showGraphPrompt`, `tensorviz.codeFocus`, `tensorviz.mcpPort`, and the deprecated `tensorviz.graphSurface`, are in the [settings reference](../docs/reference.md#extension-settings).

The selected project Python runs the model. A separate compatible tool environment
runs the tensorViz service and dependency installer; changing model Python keeps
that service running. The backend launch token stays in
VS Code; the browser receives a limited session credential. The MCP endpoint has no
bearer credential and relies on its loopback bind and Host/Origin checks.

## Support

Development versions, the service minimum, and the capture minimum are in the
[versions table](../docs/local-setup.md#versions). Capture runs in the Python
environment you select. Expression-level source edits need source positions that
Python 3.10 does not record.
Managed runtimes are available for
macOS Apple Silicon, Windows x64 and Linux x64; other architectures need a
compatible existing Python environment.

Desktop **Remote SSH** is supported too: install tensorViz on the SSH host and
select Python there. For a coding agent on your desktop, use **Forward MCP to This
Desktop**. See [SSH setup](https://tensorviz.dev/install.html#remote)
and [platform details](https://tensorviz.dev/install.html#compatibility)
for runtime requirements and other environment limits.

TensorViz's own code uses the MIT license. The bundled tldraw SDK and assets
retain their separate licenses. `LICENSE` and `THIRD_PARTY_NOTICES.md` ship in
the extension. The source is
[tensorviz-org/tensorviz](https://github.com/tensorviz-org/tensorviz).

[Getting started](https://tensorviz.dev/install.html#first-graph) ·
[Questions and bug reports](https://github.com/tensorviz-org/tensorviz/issues) ·
[Security](../SECURITY.md)

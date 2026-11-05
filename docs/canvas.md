# Explore the graph

Start with a model open in TensorViz. The Canvas lets you explore its structure without rerunning it every time you change the view.

![Annotated Canvas showing the model layers and the fit control](images/explore-graph.png)

**1 — Layers:** follow the connections through the model. **2 — Input:** the example shape comes from the source. **3 — Fit:** bring the graph back into view. This example is a source preview; shapes are recorded only by a capture.

## Start with these controls

| To… | Use… |
| --- | --- |
| See the architecture | **Layers** |
| See the captured operations | **All operations** |
| Look inside a module | Expand it, or enter it and use the breadcrumb to go back |
| Find a layer | **Find a layer** or **⌘/Ctrl+P** |
| Explain the visible components and flow | **Explain this view** (uses your configured agent) |
| Change the example inputs | **Inputs** |
| Capture again | **Run model** / **Run again** |
| See more controls | **Expand toolbar** |

Pan and zoom to move around. Turn on [code focus](code-focus.md) to reveal a selected layer's source beside the graph.

## Read the shapes

**IN** and **OUT** rows show the tensors recorded for that operation. **Not recorded** means TensorViz does not have a shape for it.

Named arguments and dictionary results keep their recorded names. Unnamed values use numbered labels; see [input and output names](capture-boundary-names.md).

A captured graph describes the inputs and path from that run. Expanding, searching, or changing the layout does not execute your model.

## Find what you need

Search by a name such as `conv1`, a type such as `type=Conv2d`, or a shape such as `shape=256`. Choose a result to jump to it. Hits inside a collapsed module stay grouped there until you open it.

## Understand a node

Use the model name at the top to switch between PyTorch model classes discovered
in this project. Search by class name or file. Models found by reading code do not
need an environment setup or a forward pass to appear in the list. **Captured**
means TensorViz has execution evidence for the displayed model; **From code**
identifies a class found without running it. Opening a class uses the normal graph
and setup flow. File and setup utilities remain under their own disclosure.

Cards keep the same content at every zoom level. Full class and function names wrap;
long names can use a wider card. Click a node to inspect it without changing the camera.
Constructor arguments stay in **More details**. The full list grows with its content
and scrolls with the surrounding selection panel.
Input and Output cards reserve only their visible rows; an empty output is header-only.

Select a node to see its callable, source and available inputs and outputs. When the
declaration can be resolved, its author's docstring appears as **Source documentation**.
**More details** includes the declared signature and complete available docstring.
This works for custom classes and functions without an agent or catalogue entry;
standard-layer reference descriptions are a fallback. A small code icon means the call was
read from code; its hover explains that tensor shapes have not been recorded.

With an agent connected in chat, selecting one node automatically requests a short
source-based explanation in the selection panel. Graph cards retain their technical names.
**Agent interpretation** distinguishes that explanation from recorded graph facts.
Open **More details** for the complete call, steps, uncertainties and linked sources.
**Discuss in chat** opens the existing chat with the current selection available.

Choose **Explain this view** in the toolbar to explain the nodes currently visible in
the Canvas, including input and output boundaries. The click captures a scope; panning,
zooming and opening a component do not recursively request more explanations. Click
again to explain the new view. Collapsed and off-camera nodes stay outside that pass.
Readable roles, a short purpose sentence and input/output meanings appear directly on
the cards, while technical names and tensor evidence remain available.

A collapsible **This view** explanation describes how the displayed components work
together. Source-linked callouts explain shared behavior such as conditioning, default
values or branches and select their existing components. Connection labels explain
existing directed relationships; generated content cannot create graph edges or model
operations. Source structure remains distinct from observed tensor flow. Follow a
callout source link to select its component and reveal the matching verified source.

The pass reads bounded batches of graph facts and shared source excerpts; it never
runs the model or changes its wiring. Progress shows completed components with a stop
button. Completed work remains after partial results or cancellation. The configured
agent is checked directly without opening or warming chat. Select a component for
implementation steps, actual upstream/downstream links, source citations and concepts.

The details control offers **Hide explanations** / **Show explanations** without an
agent call. Each pass covers up to 128 visible nodes, with a notice for larger views.
Unchanged node interpretations and complete view explanations are reused for the same
revision, agent and model; a changed scope can require a shared view explanation.
Source/capture/provider changes invalidate reuse. Saved graph files retain technical
source facts; image exports include enabled node interpretations and connection labels.

Card dimensions follow their wrapped content, including long input/output names,
tensor shapes and source links. Missing internal details do not produce a
**Contents unavailable** placeholder. Actionable errors and capture limitations remain
inspectable. Adding annotations reserves space in the displayed layout while preserving
the camera, selection and saved graph positions. Connection lists, signatures,
docstrings and parameter lists fit their content inside the existing details panel;
they do not introduce separate scroll areas.
The selection overlay scrolls within the available Canvas area, including when
the editor gives the graph only part of the window. Source-reading activity is
shown separately from tensor evidence; missing runtime shapes never start a loader.
If the host does not acknowledge an explanation request or stops responding, the
activity ends with a retryable explanation message. Existing source facts and
completed explanations remain available. No forward pass is needed for this step.
If a pass cannot start, its details show the reason rather than a zero-completed
count. Source and definition links align with the rest of a module's metadata;
long paths and route endpoints wrap without being shortened.

Explanations read source without running the model or changing your files. They are
local to the current session and refresh after source or capture changes. Saved graph
files and SVG exports retain full technical names.

## Ask for a change

Select the relevant layers and describe your change in Canvas chat. Choose your agent, model, and supported effort in chat settings; start a new chat to switch agents. [Review the proposed Python changes](source-editing.md) before applying them.

After a matching capture, native and ACP Canvas agents automatically receive [compact model context](model-context.md). Current facts can be reused without another capture; missing measurements and stale source stay explicit.

## Make it comfortable

Use the sun/moon control for appearance, or **More tools → Display settings → Appearance** to follow your editor. Layout and appearance changes keep your capture.

For a saved snapshot, use [Export and share](graph-files.md). For changed Python, use [Graph diff](graph-diff-review/README.md).

<details>
<summary>More Canvas controls and technical details</summary>

[Full Canvas reference](reference/canvas.md) · [Block families and symbols](canvas-block-families.md)

</details>

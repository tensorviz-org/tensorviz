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

Cards keep the same content at every zoom level. Full class and function names wrap;
long names can use a wider card. Click a node to inspect it without changing the camera.
Constructor arguments stay in **More details**, where the full list can be scrolled.
Input and Output cards reserve only their visible rows; an empty output is header-only.

Select a node to see its callable, source and available inputs and outputs. When the
declaration can be resolved, its author's docstring appears as **Source documentation**.
**More details** includes the declared signature and complete available docstring.
This works for custom classes and functions without an agent or catalogue entry;
standard-layer reference descriptions are a fallback. **From source** means the call was
found in code; tensor shapes remain unknown until recorded by a capture.

With an agent connected in chat, selecting one node automatically requests a short
source-based explanation in the selection panel. Graph cards retain their technical names.
**Agent interpretation** distinguishes that explanation from recorded graph facts.
Open **More details** for the complete call, steps, uncertainties and linked sources.
**Discuss in chat** opens the existing chat with the current selection available.

Explanations read source without running the model or changing your files. They are
local to the current session and refresh after source or capture changes. Saved graph
files and SVG exports retain deterministic technical names.

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

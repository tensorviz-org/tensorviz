# Open a graph

**Open as Graph** resolves setup, shows the source preview, constructs the model, and captures one call on the saved device. You do not click Run for that first capture. A matching saved profile, or a complete source recipe, starts the capture directly.

Libraries and the setup form are separate pages: [model libraries](graph-libraries.md), [setup and inputs](graph-setup.md). What the capture does and does not prove: [concepts](concepts.md).

## Contents

- [Steps](#steps)
- [After the first capture](#after-the-first-capture)
- [Discovering Open as Graph](#discovering-open-as-graph)
- [Reopening after a code edit](#reopening-after-a-code-edit)
- [Browsing models](#browsing-models)

## Steps

1. Open a folder that contains the model.
2. Run **Open as Graph** from the CodeLens above the class, the Models view, or **tensorViz: Open as Graph**.
3. Choose Python if asked. TensorViz asks before it installs libraries. See [model libraries](graph-libraries.md).
4. If constructor values or example inputs are missing, the Canvas shows **Choose a setup method**. In VS Code the recommended button is **Set up with agent**. In the plugin browser it is **Copy setup request**. **Set up manually** opens the form. See [setup and inputs](graph-setup.md).
5. Saving a complete setup writes `tensorviz.toml` and captures one forward call. The sidebar shows that call. Unknown input dimensions stay empty until something measures them. A source preview does not substitute `1 × 3 × 224 × 224`.

**Inspect source only** does not execute the model.

For eligible models with unknown inputs, a constructor-only worker can look for input sizes on CPU before the capture. The Canvas shows a compact execution strip: device, machine, and a disclosure arrow, without run-status text. Open the strip for progress, Cancel, usage, available memory, and the planned call count. Closing the details keeps them closed through later updates.

## After the first capture

**Run model** opens a review where you can choose another device, then **Run**. **Run** closes the review while the toolbar shows progress and Cancel. A model the editor auto-opened waits for that review instead of running immediately.

**Model settings** names the resolved `tensorviz.toml` profile. Mode reads `mode`, Gradients reads `grad_enabled`, and the Inputs table reads the saved recipe. **Edit** beside an input selects that field in the editor. Saving a model or tracked dependency automatically validates the open model by default. Set `tensorviz.traceOnSave` to `false` to keep execution manual; **Run again** reviews the saved values.

After a run finishes, fails, or is cancelled, **Run again** requests a fresh review. Constructor-only preparation uses **Prepare again**. The repeat waits until the previous host operation has settled.

`tensorviz.executionCache.mode` defaults to `on`. `shadow` still runs the model every time and only records whether an identical earlier run in this window could have been reused. `on` shows that earlier result after a brief Python safety check when the saved sources, configuration, inputs, seed, environment, and tensorViz backend match, on CPU. The toolbar labels a reused result. **Run fresh** in the execution details executes anyway. **tensorViz: Clear Execution Cache** drops the stored results. They stay in memory for this window and are never written to disk. Agent (MCP) runs always execute. A reused result is rewritten onto the documents' current versions, so code focus and source review follow the editor.

Run and Run fresh request graph capture only; latency profiling and training-step diagnostics are not available in this build.

These choices apply to one run. The graph describes the chosen inputs and mode, not every possible execution.

## Discovering Open as Graph

You can click **Open as Graph** in Canvas even when no Python file is selected. If a folder is open, TensorViz scans the workspace with its static model discovery and offers a searchable list of classes and paths. In a multi-folder workspace, each choice also names its folder. You still select the model, including when there is only one. Discovery does not import or execute Python.

**Choose a Python file…** browses for a model the scan did not list. The per-file class chooser handles that selection, including unverified subclasses. An empty window opens the file picker directly. Escape or cancelling the picker opens nothing. **Open Project** stays a separate action. A discovery failure shows an error rather than claiming that no models exist.

When the active saved Python file contains a discovered model, a notification offers **Open as Graph**. The scan does not import or execute the model. Several classes in the file ask which one to open. The action stays bound to the offered file if you change tabs before clicking.

The notification is on by default and appears at most once per file per window. It waits for discovery and only offers the active file in a trusted workspace. Dirty files, virtual documents, and files without a discovered model are skipped. A failed discovery does not mean the file has no models. CodeLens and the command remain available.

**Disable graph prompts** turns the notification off. The setting is `tensorviz.showGraphPrompt` (**Tensorviz: Show Graph Prompt** in the Settings editor). Turning it back on restores offers for files not already offered in this session. CodeLens, Models, and the command palette keep working while prompts are off.

The CodeLens above an `nn.Module` class, the Models view, and **tensorViz: Open as Graph** are the same opening flow. The lens stays available when that class is not already the graph on screen.

## Reopening after a code edit

After you save a compatible edit, **Open as Graph** captures the updated code with the existing saved setup. It also retries failed or cancelled openings. Reselecting an unchanged, current graph keeps its measurements and view. Showing or hiding the Canvas does not start a capture.

Edits to the model or its tracked imports invalidate the previous capture before the source preview refreshes. Tracking includes Python files the runtime verified, even when static analysis did not reach them. Reopening checks current source bytes, not only the selected file and class. Unsaved files still need a save. Incompatible inputs or invalid forward code still fail in a way you can act on. A compatible edit does not rewrite `tensorviz.toml`.

## Browsing models

The **Models** view mirrors the source tree: expand a folder, then a Python file, then select a class to **Open as Graph**. Only paths that contain discovered `nn.Module` classes appear. Folder and file rows show how many models sit under them. Class rows show the declaration line. The tooltip has the full path and bases. Folders sort before files. Names are alphabetical inside each level.

Folders and files start collapsed. Expansion survives index refreshes. A single-folder workspace starts at its contents. A multi-folder workspace shows a named root for each folder that contains models. The same path and class name in two roots stay separate and open their own files.

Grouping uses the paths the class index already has. It does not import Python, guess architecture categories, or infer which classes instantiate other classes. Saving Python or **tensorViz: Refresh models** updates the tree.

Opening a class inside a nested project uses that project's root for source discovery, imports, `tensorviz.toml`, generated setup helpers, and capture. The sidebar and **Open Canvas in Browser** share this identity. Source links still open the original files in the containing workspace.

Starting at the model's directory, the host walks toward the owning workspace and stops at the first directory that contains `tensorviz.toml`, `pyproject.toml`, `setup.py`, `setup.cfg`, or `.git`. A directory whose name cannot be a Python module component, such as `tensorviz-clean`, is also a boundary. The walk never goes above the owning workspace. An unmarked tree of valid Python directory names keeps the workspace root. A file opened without a workspace keeps its own directory.

For an independent nested project, put `tensorviz.toml` or Python project metadata at the import root you want. The `graph_ui` and `mcp_diagnostics` examples include a minimal `pyproject.toml` for this. They declare no dependencies or input values. An invalid or linked setup file still establishes the boundary, so the usual setup checks report it instead of switching to an outer project.

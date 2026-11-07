# Graph files, walkthroughs and discussions — reference

<a id="graph-files-walkthroughs-and-discussions"></a>

For the short walkthrough, see [the user guide](../graph-files.md). This page keeps the detailed behavior and implementation notes.

## Contents

- [Export and exchange](#export-and-exchange)
- [Explore the snapshot](#explore-the-snapshot)
- [Share a specific view](#share-a-specific-view)
- [Guided walkthroughs](#guided-walkthroughs)
- [Discussions on layers and connections](#discussions-on-layers-and-connections)
- [Update the architecture while keeping notes](#update-the-architecture-while-keeping-notes)
- [Compare two snapshots](#compare-two-snapshots)
  - [Matching and evidence](#matching-and-evidence)
- [File format and limits](#file-format-and-limits)
- [Build and serve the viewer](#build-and-serve-the-viewer)

A `.tensorviz.json` file contains an architecture snapshot and optional walkthrough
notes and discussions. It preserves collapsed contents, connections, public
parameters, available shapes and layout. The architecture is read-only; annotations
are editable. Nothing updates from Python automatically or executes the model.
Project frames and read-only snapshots cannot be ungrouped; expanding them changes only the view.

**Version 4** files contain canonical captured
computation. They preserve every retained operation/call, ordered argument and output
tree, typed ports, tensor/state identity, aliases, mutation effects and coverage.
Select a captured layer and choose **View captured details** to inspect those facts
entirely offline. The header distinguishes a complete capture of one execution path
from a partial capture; neither certifies other inputs or branches.

Live **Export graph**, file import, and offline inspection
are available. Export prepares the exact displayed capture through the connected
VS Code host without executing the model. Retargeting, eviction, disconnection or
invalid/private evidence refuses the export. Version 4 comparison uses canonical
arguments, tensor uses, sharing and state, with explicit unresolved correspondence. Static/legacy
export, import, notes and comparison retain their existing behavior.

Legacy snapshots preserve trace capture limits. A **Depth-limited** block has no captured
internal trace contents and may carry measured boundary shapes if it was observed;
it does not mean the Python module has zero layers. **Contents unavailable** disables expansion when no
members are present, including when restoring a saved open view. Older empty
trace snapshots show the generic contents-unavailable message because their
capture depth is unknown. Existing captured members still expand locally, and
repeat counts still refer to the same canonical representative.

**Measured boundary** describes shapes observed at that node for the traced input,
not validation of every child. Static, declared, unobserved and stale evidence
remain distinct. Export retains bounded-capture metadata, shapes and repeat
identity; executable custom Python and internal import/source paths remain in the
workspace and are excluded by the existing portable-file privacy rules.

Snapshots also preserve optional exact `parameterCount` and complete
`outputTensor` metadata (shape, dtype and logical bytes). Zero is a known value.
The byte overlay is an output tensor footprint, not peak memory; see
[overlay evidence and limits](canvas.md#overlays). Stale exports
remain uncoloured. Versions 1, 2 and 3 still open; older files lacking these
fields retain their shapes without inventing counts or assuming float32 bytes.
The importer validates the optional fields and retains only the closed tensor
metadata subset, preserving existing source and private-path filtering.

## Export and exchange

1. In Open Canvas in Browser, choose **Export graph** after the model finishes loading.
2. Choose **Open graph file viewer** and open the downloaded file.
3. Add notes or replies, then **Download graph file** and send the updated file.

Recipients need the graph file and a static viewer page, without Python, Torch,
VS Code, an account or an API key. Your localhost URL is not a share link. Viewer
comments travel in files: the viewer has no online storage, simultaneous collaboration
or automatic merging of separately edited copies. Use one exchanged copy when
collecting replies.

The VS Code Canvas offers **Add to tensorViz** in Export's place: it uploads the same
file to your tensorViz account, private until you choose who can see it on
tensorviz.dev; see [Add graphs to tensorViz](../share-to-website.md).

Notes remain in the tab until downloaded. Invalid files preserve the current graph
and notes. Opening another file asks before discarding edits; reload/close triggers
the browser's unsaved-change warning. A newer file choice cancels any earlier pending
replacement. Post or discard draft messages before downloading.

## Explore the snapshot

Navigation, both searches, metric overlays and the connection guide share one
vertical controls column. Results and legends move the controls below them;
the column scrolls when space is limited, without resizing the graph viewport.
**Fit**, zoom, **Blocks** and **Expand all** change presentation only.

**Search layers** finds a layer or module by name/type; choosing a result or
pressing Enter reveals its containing modules. Escape clears this search.
**Search the graph** supports [structured queries](canvas.md#search);
its results select the visible containing block without changing expansion.
**Parameters** and **Output tensor bytes** retain the recorded evidence and
legend limits described above. Saved views and discussions remain available
beside these controls, and architecture stays read-only.

Focus and search controls derive labels only for the targets they display. Their render-local node lookup includes hidden nodes, so searching inside collapsed modules keeps the same results without opening those modules.

## Share a specific view

Choose **Saved views**, select a layer, block or real connection, and arrange the
canvas. **Share this view** stops any camera motion, captures the view and makes
it the opening view. Give it a title and optional note, then **Download graph file**
and send that file.

The recipient opens at that view, with its selection, expanded modules and camera.
The camera keeps the same center on a different screen size. Named views can be
revisited, reordered, updated or deleted; **Make opening view** chooses which appears first,
and **Clear opening view** returns to the usual graph overview. Changed or missing
targets remain marked after an architecture update. These are views inside a file,
not hosted links. Walkthroughs and discussions travel alongside them.

## Guided walkthroughs

Choose **Walkthrough**. Select a layer or block, arrange and expand the graph, then
**Add current view**. Edit its title and note; **Update saved view** recaptures the
selection, positions, expanded blocks and camera. Move steps up/down or delete them.

Readers use **Start walkthrough**, **Previous** and **Next**. Each step restores its
saved view. They can pan and explore freely, use **Show saved view** to return, or
**Exit walkthrough** to continue exploring. Notes are plain text, including line breaks.

## Discussions on layers and connections

Select a layer and choose **Discuss layer**, or click an expanded connection and
choose **Discuss connection**. The **Discussions** panel also lists a selected
layer's connected edges. Display-only edges created for collapsed modules cannot
anchor a discussion; expand the module or select a connection from that list.

Choose **Start discussion**, enter a name and message, then **Add message**. Readers
can **Reply**, **Resolve discussion**, **Reopen discussion**, and **Show current target**.
Names are unverified labels, not accounts. A draft keeps its original target even
if you select another layer. Download and exchange the updated file to share replies.

## Update the architecture while keeping notes

**Update architecture** takes architecture from a newer graph file and retains the
current walkthrough and discussions. The replacement must contain no annotations;
annotated replacements are refused so neither set of notes is silently discarded.

Discussion anchors keep their original snapshot date and semantic context. Parameter,
module-content or connection changes are marked **Context changed**. Deleted targets
are **Target missing**; non-unique connection matches are **Target ambiguous**. Original
details remain available. Layout, expansion and edge ordering alone do not change
context. Renamed layers are not guessed or rebound. Walkthroughs also flag changed or
missing targets; review the notes before updating a step's saved view.

## Compare two snapshots

Open the newer file, choose **Compare**, then **Choose Before file** to select the
older snapshot. For version 4 captures, the report separates computation changes,
exact tensor-use changes, runtime observations and execution conditions. Missing
correspondence or partial evidence cannot certify equality. Static/legacy comparison
lists added/removed layers and connections, parameters and other structure.

**Show before** and **Show after** focus a changed layer or connection at a readable
scale beside the comparison panel. **View before/after** switch the canvas version. Its header always labels the side.
Canonical switching follows a corresponding selected card or destination operand slot,
even if capture IDs or the operand's producer changed. Without a confident counterpart,
the destination restores its own view. **Inspect before/after** opens exact retained
records without switching the canvas, including module records and large argument trees.
Previews are limited to 4,000 characters; comparison uses the complete facts. Each
section initially shows 25 records, with an explicit **Show more records** action.
Closing comparison returns to the current After graph, preserving its notes and
message drafts. Invalid comparison files keep the previous comparison available.
Comparison is temporary and read-only; neither file's annotations are merged.

### Matching and evidence

Comparison matches portable layer IDs, which retain source hierarchy when available.
It compares operation, kind, parameters, module membership, ports and connections.
Connection order and generated edge IDs, layout, annotations and export dates do not
count as architecture changes. Duplicate connections are counted individually.

Unmatched IDs remain added/removed layers. Structurally identical unmatched layers
are flagged as uncertain possible renames; tensorViz never silently pairs them.
Different model names also produce a warning. Recorded shapes and tracing status
are separate evidence: different inputs or tracing availability can change those
without changing the architecture. Comparison does not execute either model.
This matching describes versions 1–3. Capture-scoped event IDs must not be matched
using those legacy rules. Version 4 uses conservative source/ownership/dataflow
correspondence and exact argument/value-use comparison. Repeated operations and numbered
containers can remain unmatched: unmatched does not mean added or removed. A matching
captured computation does not prove numerical equality or behavior on other inputs.
Mixed canonical/legacy files report comparison unavailable. The rules for a
captured computation are in [coverage and correspondence](graph-review.md#coverage-and-correspondence).

## File format and limits

Version 3 uses `format: "tensorviz.graph"`, `version`, `snapshotId`, `exportedAt`,
`status`, `warningCount`, `document` and `collaboration`. The document is a validated
portable subset of GraphDocumentV2. Collaboration contains one ordered walkthrough,
discussion threads, named saved views and an optional opening-view id. Version 1 and
2 files remain readable; older viewers reject version 3 instead of dropping saved views.
Saving notes preserves the architecture export time. Saved views retain their original target context, selected nodes or a
real semantic connection, camera, expansion and optional capture viewport dimensions.
The opening-view id must reference a saved view in the file. Snapshot labels identify
context and do not authenticate a file.

Version 4 adds `capture: PortableCapture` v1, versioned independently from the file.
Optional initializer-entry scalar snapshots survive export/import and appear as
observations in comparison. Older captures have no such evidence. Opaque initializer
objects carry no payload or type name and do not invalidate complete forward
capture. Sensitive numeric parameter names and path-like variadic keys refuse
lossless export; benign declared parameters such as `Embedding._freeze` remain
portable.
Its document is checked against the payload before rendering: all captured events,
exact operand paths and producers, original edge/port identities, invocation ownership,
tensor facts and declared metrics must agree. Source paths become deterministic
aliases, source bindings follow them, and live workspace/editor identity is removed.
Recognized private/path-like argument literals refuse export/import instead of losing
an operand. Version 4 does not apply legacy parameter filtering to captured arguments.
Its display-only summaries are derived separately from retained evidence. The same
Python-generated structural validators are precompiled at build time, so the viewer
keeps its existing CSP without `unsafe-eval`. Relational/privacy checks share a corpus
of acceptance and refusal cases with the Python validators. Imported evidence is not authenticated.

Captured discussions include exact record/port/value context, including literals and
sharing, within the existing anchor-size bound. A changed argument invalidates that
context; layout and notes do not alter the retained computation. Large target contexts
can exceed the existing note-anchor bound and must be narrowed to a smaller target.

Imports are limited to 96 MiB, 20,000 nodes, 50,000 edges, 64 hierarchy levels,
8 parameter nesting levels, 50 walkthrough steps, 50 saved views, 200 threads and 1,000 messages.
Notes/messages allow 4,000 characters each. Invalid identities, parents, coordinates,
unsafe keys and unsupported versions are rejected before rendering.
The importer checks every parsed own key, including ignored metadata and array
entries, before consuming the file. Ordinary shallow JSON avoids a reviver callback
for every scalar; unsafe keys or unusually deep containers use the original
reviver traversal, preserving its rejection behavior and engine-dependent depth
handling. This optimization does not change schema or privacy validation.
Version 4 permits up to 32,768 declared ports per node within the same total file
limit; versions 1–3 keep their 64-port bound. Runtime capture budgets remain independent
from file size, document size and display collapse. Limits refuse without truncation.

Source text, local paths, arbitrary metadata, credentials, live-session identities,
weights and data files are excluded from architecture export. Internal/source/path
parameter fields and absolute local path values are omitted. Ordinary public values,
labels, names and annotation text remain visible: review them before sharing.
Static/traced/stale status describes the original snapshot, not a new runtime check.
Stale live graphs export as stale. Canonical graphs require a live connection to
retrieve their retained capture; disconnected static/legacy graphs export as stale.
Portable preparation accepts at most 16 MiB of retained input and transfers at most
16 MiB of detached capture. One preparation runs per host; the browser waits at most
15 seconds. Oversized payloads refuse without truncation.

## Build and serve the viewer

From the repository root, run `npm run build:viewer --prefix canvas`. Serve `canvas/viewer-dist/` with a static
HTTP server, for example `npx --yes http-server canvas/viewer-dist -p 4175 -c-1`. It includes
local tldraw fonts/icons/translations and makes no backend or CDN requests. Files
are read locally in the browser. This is a static website build, not a self-contained
HTML export. The extension bundles the same viewer under `media/app/viewer/`.

JAX exports keep inferred shape provenance. Import refuses a file that calls those
shapes measured. Large graphs lay out with iterative ranking above 1,000 nodes or
4,000 edges so the layout does not recurse without bound. All nodes and edges are
retained. See [licensing](../licensing.md) before you ship a viewer build.

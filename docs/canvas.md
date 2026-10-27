# Canvas

## Contents

- [Automatic readiness and one starting screen](#automatic-readiness-and-one-starting-screen)
- [Starting-screen appearance](#starting-screen-appearance)
- [Connected empty and recovery states](#connected-empty-and-recovery-states)
- [Unexpected Canvas errors](#unexpected-canvas-errors)
- [Integration with semantic captures](#integration-with-semantic-captures)
- [Module input and output shapes](#module-input-and-output-shapes)
- [Commands](#commands)
- [Boundaries and ownership](#boundaries-and-ownership)
- [Visibility, disposal and replay](#visibility-disposal-and-replay)
- [Placement and connections](#placement-and-connections)
  - [Active toolbar layout](#active-toolbar-layout)
- [Controls](#controls)
- [Module tiles](#module-tiles)
- [Source controls](#source-controls)
- [Canvas appearance](#canvas-appearance)
- [Search](#search)
- [Overlays](#overlays)
- [Editor shape hints](#editor-shape-hints)
- [Module focus](#module-focus)
- [Operation input and output shapes](#operation-input-and-output-shapes)

The sidebar Canvas, then navigation, search, overlays, and appearance. Opening a model is [open a graph](graph-opening.md).

The persistent **Canvas** receives **Open as Graph**, per-class CodeLens, and Models selections in the `tensorviz.sidebarCanvas` webview, initially in the `tensorviz-canvas` secondary-sidebar container. The editor version is in the [versions table](local-setup.md#versions). The view stays wherever you move it.

## Automatic readiness and one starting screen

In a trusted window the home automatically checks only tensorViz's owned runtime.
A ready runtime shows **tensorViz is ready** regardless of VS Code's selected
project Python or its libraries. Home does not discover environments, inspect
capture metadata or offer a project Python picker. Missing/broken tool runtimes
offer **Set up tensorViz**, **Check again** and **Show log**
where relevant. Workspace Trust has its own recovery action. Opening home never
downloads, changes dependencies, saves a Python choice or executes model source;
checking an installed tool runtime may run its existing cached application probe.
Model environment choices, capture-library conflicts and model-specific libraries
are checked after the user selects a model, in the existing opening recovery UI.
A ready service does not certify a model import or capture. Checks abort on
disposal and discard cancelled results; changing project Python no longer changes
home readiness. Cached model-list updates still cannot replace a retained graph.

Service setup keeps the same home visible. Its renderer iframe connects hidden
and appears only after a model, project or PR review is selected, avoiding a
second **Explore your model** screen. Explicit openings continue through runtime
setup; failed home actions keep their host-owned model identity for **Retry opening**.
Runtime setup does not replace that retained request. Downloads and dependency
installation keep their existing approval; project Python does not gate Open PR.

## Starting-screen appearance

The home screen is already an HTML webview, independent of the running graph
renderer. Its empty state now has a dotted background and an illustrative neural
graph with slow animated connections, the tensorViz mark, a prominent **Open as
Graph** action and a secondary **Open Folder** action in an empty window. The
illustration is decorative; it is not a preview or measured capture of your model.
Cached model rows replace the illustration once discovery supplies choices.

Colors and typography follow VS Code's theme, including light, dark and high
contrast. Narrow layouts wrap text without horizontal scrolling. Animations run
without playback buttons. An
OS/browser reduced-motion preference disables motion. Hidden documents pause
the animation; becoming visible resumes it when reduced motion is off.

The graph is inline SVG, animated with CSS; local JavaScript handles actions and
visibility and reduced-motion handling. There are no remote fonts/assets, new runtime dependencies or
additional backend startup. The existing nonce-based script CSP and host-owned
model-index routing remain in place.

## Connected empty and recovery states

The renderer's own empty Canvas offers **Open as Graph** and **Open PR**
in the same neural-graph layout, using the existing Paper/Ink themes. Open as Graph
asks the native host to run the existing model picker; the webview cannot supply
an arbitrary command or model URI. Opening the screen itself does not select or
execute a model. Animations have the same visibility and reduced-motion behavior
as the local home; it is hidden behind home during native onboarding. Connecting/disconnected empty views show truthful host status
and recovery instructions instead of unusable model actions.

Ended sessions and standalone empty-model views use the same renderer layout.
An empty source review keeps its coverage notes and explains that no affected
model was identified. **View code changes** opens the corresponding native diff;
**Return to Canvas** closes the review through the existing controller. Standalone
views offer source-reference copying through their existing contract. Native
service preparation and backend failure screens preserve the current messages
and make long details scrollable.

## Unexpected Canvas errors

The Canvas, browser view and run comparison share one top-level error boundary,
`canvas/src/ide/CanvasErrorBoundary.tsx`, applied in `main.tsx`. Before it,
any uncaught rendering or effect error unmounted the whole renderer, leaving a
blank (black in dark themes) view; the tensorViz log only showed `canvas NOT
MOUNTED`. The view now shows **The Canvas stopped after an error** with the error
message and a collapsed **Error details** section, containing the stack and
component stack (up to 8,000 characters), for bug reports. The error is also
written to the webview console.

**Reload Canvas** remounts the app in place. In VS Code, the new mount sends
`canvasConnect` again, and the host answers with the current state, as for a
recreated view. The page itself is not reloaded, so the launch token that startup
removes from the URL is kept. Source files and saved setup are unchanged. React
boundaries do not catch errors thrown in event handlers or asynchronous callbacks.
Narrower boundaries still take precedence: a setup draft the form cannot show
stays inside the setup panel, and the graph keeps working.

## Integration with semantic captures

The sidebar uses the same canonical capture reader, portable export preparation,
candidate controls and attempt ledger as the current browser surface. It can
inspect and export a captured model without opening an external browser session.
The native bridge accepts these typed requests with its existing session and
view-identity checks; stale targets, evicted captures and cancelled exports still
refuse a result.

Native and browser source reviews have separate ownership registries over the
same source-edit controller. Browser reconnect/disconnect cancels browser-owned
reviews without cancelling a native review or MCP work. Candidate and review
responses return to the originating surface. Native ownership is checked against
the controller's reserved source request, not a renderer-supplied request ID.

Validation summaries and reconnect snapshots come directly from the attempt
ledger, including capture attribution. The sidebar retains those summaries and
never reconstructs measurement status from a terminal receipt or a changed graph
revision. An external browser connection change cannot cancel a native export.

## Module input and output shapes

At inspection zoom, collapsed model module cards show **In** and **Out** rows
below their capture/evidence badges. For example, the TinyTransformer `encoder`
and `norm` cards show the captured `(1, 10, 32)` shape and `float32` dtype in
both directions. These rows also appear in the ReactFlow module renderer.

The rows use the module call's canonical `capturePorts`, not the synthetic child
ports used to route connections through a collapsed card. Multiple inputs or
outputs retain separate rows with their recorded dictionary keys or keyword names.
Unnamed values use concise numbered labels; see [input and output names](capture-boundary-names.md). Scalar tensors
display `()`, missing tensor metadata remains explicitly unknown, and non-tensor
arguments show their recorded kind. An empty captured interface stays empty.
Legacy snapshots use the existing provenance-aware single-tensor shape fallback;
stale evidence retains its **Stale shapes** badge.

The shared module height calculation reserves space for every row before laying
out sibling cards and expanded parent frames. Compact unchanged-input calls and
expanded frame headers keep their existing presentation.

## Commands

| Action | Behavior |
| --- | --- |
| Open as Graph / CodeLens / Models | Exact `{ uri, cls }` selects that class and reveals the existing Canvas. An unnamed class keeps the existing project-opening behavior. |
| Open Project as Graph | Selects the requested workspace folder in the same Canvas; repeating it preserves the existing project state. |
| Show Canvas / Open Canvas | Reveals the retained model/project, or the local starting screen. Does not prepare a service or infer a target from the active editor. |
| Hide Canvas / Toggle Canvas | Uses VS Code's native action to hide this specific view while keeping it contributed. Checkpoints and the controller survive; Show/Hide labels follow native visibility. |
| Open Canvas in Browser | Opens the controller's selected model or project using the existing authenticated browser session, without resetting projection, review or validation. An empty Canvas explains how to select a graph and opens nothing. |

The editor-title toolbar and overflow expose Show Canvas or Hide Canvas. The
Canvas title offers Hide Canvas and Open Canvas in Browser. The existing toggle
shortcut is unchanged: **Ctrl+Cmd+G** on macOS, **Ctrl+Alt+G** on Windows/Linux.
Native sidebar resizing, moving and maximizing remain VS Code-owned. No other
extension is moved, closed or its layout reset. Explicit Hide marks only this
view hidden in its current container; Show Canvas restores it. VS Code may hide
an empty container's navigation entry after the user's explicit Hide action.
Ordinary installation has no graph-request condition hiding the entry.

`tensorviz.graphSurface` is deprecated. Remove saved `browser` or `editor` values:
all values now use the sidebar, including old settings. Use the explicit browser
action for a spacious external page. It always opens the selected graph even if
another source editor has gained focus. `tensorviz.openProjectInBrowser` remains
a hidden compatibility alias to this selected-graph action; legacy folder
arguments no longer retarget it. The old model-panel serializer closes retired
panels on reload rather than restoring a competing controller. Existing live
comparison panels and the separate run-comparison view remain available.

Opening or switching source editors never replaces a selected model/project,
even with the legacy opt-in `autoOpenCanvas` setting enabled. That setting now
only provides initial opening before a target or requested Canvas exists. Source
changes to the selected model still refresh through the existing controller.

The command reveals the retained model or the local starting screen. Its
internal command argument `{ uri, cls }` prepares the service and selects that exact
class through the current controller and is available for integration tests.
Omitting it never infers a target from the active editor. Repeated selection of
the current file/class preserves the projection and run evidence when its source
is unchanged and the opening is current. After a code edit, Open as Graph reuses
saved setup and captures again; failed and cancelled openings can be retried. Showing or
hiding the retained Canvas does not execute a model. **Open as Graph** and
**Refresh graph** resolve setup and capture through the opening controller;
opt-in save tracing reuses a saved profile.

## Boundaries and ownership

| Component | Responsibility and concrete interface |
| --- | --- |
| `CanvasBridge` in `extension/src/shell.ts` | Shared authenticated iframe HTML/options and existing clipboard/export/toast services, constructed with a `vscode.Webview`, `ShellTarget` and output channel. `onMessage` can consume native traffic; `dispose` removes bridge listeners only. |
| `CanvasShell` | Editor-panel subclass retaining `panel`, `create`, `attach`, `reveal` and panel disposal. Existing live editor-only consumers and run comparisons keep their interface; retired model panels are closed by the serializer. |
| `SidebarCanvas` | One registered provider per window. `show(): Promise<boolean>` reports backend availability; `hide()` changes only the Canvas contribution context; `post(LocalBrowserHostPublish): boolean` delivers or retains state; focus returns false unless this exact native view is ready and visible. `prepareBackendRestart()` checkpoints presentation and detaches the old service; `reconnect()` mounts the current service without revealing a hidden view. `setError()` makes the next show retry startup. |
| `SidebarCanvasOptions` | `prepare()` acquires a `ShellTarget` without selecting/executing; `receive(LocalBrowserMessage)` dispatches typed intent; `replay()` requests current controller state; `viewConnection(viewId, connected)` registers actual ready/visible native presence; `channel` supplies existing shell diagnostics. |
| `LocalBrowserGraph` | Existing source projection, project expansion, validation and native review authority. `attachCanvas(post)` returns a transport-only disposable; `handleCanvasMessage` and `replayCanvas` connect the native view. `open(argument, 'canvas')` and `openProject(argument, 'canvas')` select statically without minting a browser session. `openSelectedInBrowser()` explicitly opens the retained model/project without reselecting. `setCanvasView()` binds native MCP presence. Internal browser callers retain their existing routing. |
| `nativeCanvasTransport.ts` | Adapts the existing `LocalBrowserApp` transport interface to the shell. `view=canvas` uses the same UI and source-review/validation clients as the browser, without browser bootstrap, cookies or a WebSocket. |
| `nativeCanvasPresentation.ts` | Captures presentation-only node fields, camera and selection. Restores matching graph IDs without replacing source structure, parameters or measured evidence. Prevents the initial fit from overriding the restored camera. |

The controller is the existing window-owned object, not a second graph engine.
The native transport attaches lazily on the first successful sidebar startup.
Concurrent reveals share the provider's startup promise and the existing backend
startup promise. The backend remains window-owned. A successful Python-environment
change automatically reconnects Canvas and replays its existing target. Backend
death or a failed restart leaves an actionable error; **Show Canvas** reconnects
to the current service.

Repeated **Open as Graph** command-palette discovery reuses the ready window service,
including its actual backend directory and launch identity. It must not start a
second process behind an already mounted Canvas.

## Visibility, disposal and replay

`tensorviz.canvasRequested` controls only this view's `when` clause.
`tensorviz.canvasVisible` follows `WebviewView.visible`, visibility events and
actual disposal. A requested hide is not treated as an actual visibility event.
No command closes the whole secondary sidebar, moves another view, locks an
editor group or changes VS Code layout preferences.

The provider registers `retainContextWhenHidden`. Ordinary native hiding keeps
the iframe. Because a contribution `when` change can destroy a webview view,
actual disposal removes only the view and its bridge subscriptions. It does not
retarget, cancel a run/review, dispose a controller, clear measured evidence or
stop the backend. A fresh view receives the current projection/freshness or
project snapshot and retained completion receipts. Failed hidden delivery also
causes a replay on the next visibility event. An unchanged reveal does not reload
HTML or re-fit the canvas.

The most recent validation/review receipts are retained alongside authoritative
controller state. A manual validation result is also presented through the
existing validation-state UI after recreation, so a completed run cannot silently
return to “Not run”. Selecting another model clears obsolete cached receipts. Source changes retain
outcomes against their completed revision, so they remain explicitly stale.
Presentation is keyed by the applied source projection and workspace, not a
saved-graph ID. It is limited to 4 MiB, in memory for this window only. Node and
expansion changes checkpoint immediately; camera gestures coalesce for 100 ms
and flush on pointer release, blur or page hide. **Hide Canvas waits for an
authenticated checkpoint round trip**, reading the current camera even before
the store listener has run. A newer Show/Hide request wins over an older pending
hide. An unresponsive renderer times out after one second, logs the checkpoint
failure, and retains its last checkpoint. Unexpected renderer termination can
recover only the last delivered checkpoint; full window reload persistence is
not introduced.

Restoration changes only matching presentation fields, waits for expanded child and connection
shapes before reselecting them, and suppresses the initial fit. It never replaces
source parameters or measured evidence. The sidebar strip exposes the current
validation verdict after hidden completion or recreation. At widths up to 600 px,
the toolbar wraps, the More menu stays inside the iframe, search has usable width,
and diagnostics, source forms and the minimap remain reachable. Native resizing
itself does not auto-fit the camera.

The shell checks the iframe window and exact backend/dev-server origin before
relaying requests. The provider validates the current native session, existing
wire schema and allowed view request types. Only the existing host controller
handles source writes and execution permissions. Consumed native state is not
retained in the shell log; backend exceptions are not echoed into startup HTML.
The browser broker and authenticated external opening remain intact.

## Placement and connections

New graph layouts reserve each card's rendered width and height, including
captured ports, diagnostics and review content. Expanded modules place their
children with 48px between ranks. Inputs sit beside the first visible entry;
outputs sit beside the final visible exit. This keeps boundary connections short
instead of drawing diagonal lines through a tall module.

The footer **Layout direction** button, beside Fit, offers **Auto**,
**Horizontal** and **Vertical**. It stays available when the toolbar is
minimized, and its icon shows the current direction.

- **Horizontal** runs the top level and every expanded module left to right.
  Ports sit on the left and right, and a connection that skips an alternate path
  inside a module uses a lane below the row.
- **Vertical** runs everything top to bottom. Ports sit on the top and bottom,
  and skips use a lane to the right.
- **Auto**, the default, picks Horizontal for a canvas at least as wide as it is
  tall, such as a maximized editor. It picks Vertical for a narrow sidebar.
  Resizing across that boundary changes the layout once the size settles, about
  0.2 s later. The graph then fits again. A small band around square prevents
  flipping while you drag a divider, and a hidden canvas is ignored.
- An explicit Horizontal or Vertical choice never changes when you resize.

Changing direction re-runs the whole layout, replacing manually moved positions.
Boundary cards line up straight with the module's actual first and last layer.

The choice lasts while the page is open, across models and source refreshes.
The VS Code Canvas also restores it, with the positions laid out in it, after the
view is hidden and shown. A reloaded **Open Canvas in Browser** page starts at
Auto. Captured review overlays retain the live direction and positions; source-only
reviews, legacy graph files and thumbnails keep their original mixed layout. The
Layout button is hidden during a review. Auto catches up on return to the live graph.

Named tensor ports keep one position even when a tensor fans out to both the next
layer and a skip connection. Ports follow the layout direction, not card positions,
so dragging never flips them. The live canvas, SVG overview/export, edge selection
and insert affordance use the same curve geometry, bounds and direction.

Use **Expand toolbar → More tools → Auto layout** to apply the new placement to
an existing arrangement, including the contents of expanded modules, in the
current direction. Saved views and ordinary source refreshes preserve top-level
positions.
Switching to another model starts its own layout even when the two models share
Input/Output node IDs. **Fit** changes the camera, not placement.

Saved workspace views keep their existing format; portable exports record the direction:

- A saved view opens at its saved top-level positions while ports follow the
  current direction. A view saved left to right and opened in a Vertical (or
  narrow Auto) canvas keeps its row until you run **Auto layout**.
- `.tensorviz.json` exports keep a validated optional `view.layout.direction`. The
  viewer opens the saved positions with matching ports and uses that direction for
  expansion, search and reset. Missing direction keeps legacy behavior. Older viewers
  ignore the field and draw their original mixed port sides.

This is presentation only: layout never adds, removes or reinterprets a semantic
connection, and a side lane alone does not claim residual or execution semantics.
It is not a general obstacle router for arbitrary hand-arranged graphs. Moving
cards into one another or into another branch can still create overlaps.

Captured ATen tiles use readable operation names at every zoom level and in the
SVG overview and image exports: `aten.mean.dim` is **Mean**, `aten.view.default`
is **Reshape**, `aten.bmm.default` is **Batch matrix multiply**, and
`aten._softmax.default` is **Softmax**. The exact target remains in captured
details; it is no longer repeated as a subtitle.
This changes display names only. Attention expands into its existing operation
tiles, with the same readable names, rather than an explanatory diagram.

Readable names also apply to older/imported graphs without capture metadata.
Qualified `aten.op.overload`, `torch.ops.aten.op.overload`, and `aten::op` targets
are recognized in the layer's name, ID or display name, as well as captured labels.
Common convolution, pooling, normalization, activation, indexing, embedding,
resize, reduction, tensor allocation and loss operations have concise names.
For example, `aten.max_pool2d_with_indices.default` becomes **Max pool 2D**,
`aten._native_batch_norm_legit_no_training.default` becomes **Batch norm**,
and `aten.upsample_bilinear2d.default` becomes **Bilinear resize 2D**.
Fused CPU/CUDA attention kernels use **Scaled attention**. In-place operations
retain **(in place)** and backward kernels retain **backward**. Unlisted ATen
operators get a readable fallback from their target words; custom operations,
modules and grouped layout cards keep their existing titles.

Quick-jump, graph queries, focus labels and the graph-file viewer use the same
names. Search accepts both the readable name and the original dispatcher target,
including `type=Max pool`. Captured targets, graph IDs, ports and edges are not
rewritten, and downloaded graph documents retain their exact original data.

Expanded coverage was checked in the production browser viewer with actual
PyTorch 2.7.1 CPU/eval captures. That version is the recorded corpus for this
name check, not the CPU pin in [local setup](local-setup.md#install). Contract
fixtures may still record `torchVersion` 2.7.1 for those same graphs. The corpus
is: a functional vision block (4 operations), token
embedding block (3), `nn.MultiheadAttention` (29), torchvision ResNet18 (90), and
MobileNetV2 (205). Browser checks cover every captured operation title, both
search names and export preservation. An additional browser regression covers
older imported graphs without capture references. CUDA kernels are covered by
name-mapping tests; they were not executed on a GPU.
Before/after screenshots use the same captures and camera positions:
vision before,
vision after,
tokens before, and
tokens after.
The vision block runs functional batch norm, max pooling and bilinear interpolation;
the token block runs embedding, layer norm and SiLU. These checks use the isolated
graph-file viewer, not the installed VS Code extension.

Captured graphs expose **Layers / All operations** in the header. Layers folds
module internals and safe tensor-layout sequences without running the model.
At narrow widths, header controls wrap to keep Inputs and capture actions reachable.

In **Layers**, including the initial view before choosing a detail level,
consecutive slices belonging to the same verified Python expression
appear as one **Slice** card with only its original IN and final OUT shapes. For
`self.pos[:, :tokens.shape[1]]`, this is `(1, 12, 32)` → `(1, 8, 32)` for eight
tokens. **All operations** exposes PyTorch's separate dimension-0 and dimension-1
slices. Grouping requires exact source-expression identity and an unbranched tensor
connection in the same invocation with increasing slice axes; separate expressions,
repeated executions, unavailable source bindings,
failed/stale measurements and intermediate uses stay separate. Capture records,
ports and exported semantics remain unchanged. Older captures need **Refresh** to
receive the source-expression hint. A grouped card has no combined source-edit
recipe; use Source or switch to All operations for individual operation controls.

The model name in the header opens file actions: **Export graph** and **Open graph
file viewer**. Export keeps the existing capture and freshness requirements. Escape
closes the menu and returns focus to the model button; clicking outside also closes it.

### Active toolbar layout

The toolbar stays minimized by default. Model identity and its file occupy a
readable row at sidebar widths; view, capture and Canvas tools wrap in groups
below it. **Expand toolbar** restores named Layers/All operations controls,
search, model file actions and More tools. **Run model** retains its text in both toolbar sizes, including at narrow widths.
Expanded **Inputs** also retains its label. Expanded Open PR/Review changes keep readable
labels; minimized mode uses their accessible icon buttons at every width.
On a captured graph, **Run model** opens one execution review.
Opening a model captures without that review.
The appearance switch stays reachable in compact,
expanded and review views; fullscreen uses the same controls.

PR review shows its title above the compared revisions. Native code diff, refresh
and close actions form one group beside appearance and Open PR. The affected-model
selector, Before/Diff/After modes, current change, counts and coverage disclosures
use the existing Paper/Ink surfaces and wrap inside narrow panels. Captured-context
reviews retain their compact change navigator without revision switches or a new
capture-explanation panel. Coverage limits, unresolved matches and refresh notices
retain their meanings. No graph geometry or source-comparison behavior changes.

Header, footer, menus, search and input setup share graphite surfaces, lavender
selection/focus states, consistent typography and thin borders. These chrome tokens
are scoped separately from graph/node colors. Error and warning statuses retain
their semantic colors. Popovers use opaque backgrounds, fit the available viewport,
and remain reachable in fullscreen. Search opens beside its header control. Reduced
motion disables the short menu transitions.

## Controls

- **Layers / All operations** and the **Before / Diff / After** review selector
  use a sliding selection indicator. Tab enters the selected option; Left/Right
  changes the view, and Home/End selects the first/last option. Reduced motion
  disables the slide. Switching detail never requests a model run.
  Graph view, model capture and canvas tools have separate compact groups.
- The toolbar starts minimized to give the graph more vertical room. The compact
  header shows the current class and file on the left, with icon-only **Layers**
  (inward arrows), **All operations** (outward arrows), **Inputs** (sliders),
  a labelled **Run model** (play) and **Expand toolbar** on the right. The selected
  detail level is highlighted; tooltips and accessible names identify every icon.
  Inputs opens the existing setup form, and detail buttons change expansion without
  recapturing. Hover the title to read a truncated path. **Run model** opens the
  shared review of device, inputs and workload. Narrow controls wrap when needed,
  keeping Run model and Cancel reachable.
  A capture shows **Running…** and an
  icon-only **Cancel** button. Expanding restores the full controls and saved
  profile/configuration rows. Collapsing and expanding preserve the graph and
  selection. **Collapse toolbar**, the small chevron at the header's bottom-right,
  returns to the compact row. The choice lasts for the open page and survives
  capture refreshes; a newly opened or reloaded page starts minimized. Footer status,
  source notes and zoom controls remain available. Fullscreen retains its full
  action strip, including the exit control.
- **Find a layer** opens quick-jump search. **⌘/Ctrl+P** opens the same search.
  Local Canvas results show the layer/operation name and its full module path.
  Captured module calls use their registered path; exact internal IDs remain
  searchable. The footer shows the full result count and Previous/Next pages of
  twenty results. Use ↑/↓ to select and Enter to jump; typing and paging do not
  move the graph. Escape closes search and restores the Find button focus.
  **⌘/Ctrl+Shift+P** opens VS Code's command palette even while a Canvas input
  has focus. The nested page forwards this one fixed action through the native
  shell, because keyboard events cannot bubble across its frame boundary. Alt
  combinations and standalone-browser shortcuts retain their existing handling.
  `type=Conv2d`, `shape=256`, and `attr:padding=1` queries retain the graph search's
  regular-expression matching and roll up hidden hits to visible modules. Closing
  search clears its temporary highlighting.
- **Inputs** opens the existing setup form. After capture, this button replaces the
  floating captured-input summary. The form still shows the saved inputs and setup evidence.
  It opens on the left, including on first use. Close or Escape hides it without
  dropping unsaved values; Inputs reopens the draft. The available AST graph stays
  visible while setup is incomplete, with source-only status until capture arrives.
- **Run model** opens a review of the host, device, inputs and workload. **Run**
  starts the selected plan. Source-only and stale graphs use
  this same action to recover their saved setup. **Cancel** stops the active phase.
  **Running…** remains labelled in narrow sidebars; the busy pulse respects reduced
  motion. The sheet body scrolls independently with wheel, trackpad and keyboard,
  while its header and action footer stay visible. Panel gestures do not reach
  graph controls.
- **More tools (⋯)** groups existing actions under View, Tools, Navigate and Export.
  Blocks/Expanded view, SVG overview, unchanged-call visibility, Auto layout,
  Select tool, Focus selected, zoom, fullscreen and Download SVG remain available.
  **Blocks view** collapses modules and fits the resulting graph after layout,
  including when Blocks is already selected or the graph has no modules. Both
  interactive cards and SVG overview fit the current canvas dimensions.
  Checkmarks reflect the current view, SVG renderer and active tool; disabled
  commands keep their explanations. **Display settings** contains metric overlays
  and their measurement limits. Search and overlay controls no longer occupy the
  canvas permanently.
- The footer shows capture freshness, the latest attempt, graph-node count, the
  **Layout direction** button and zoom/fit controls.
  The latest-attempt button opens a result card with status, attempt setup and
  measurement provenance. Close or Escape returns focus to its trigger.
  The count includes operations, state and input/output boundaries; it is not a
  count of model layers or measured module calls.
  **Source** appears when a layer is selected and uses the existing editor navigation.
- The graph automatically refits as the available canvas width or height changes,
  including resizing the Chrome window, dragging the VS Code sidebar divider,
  opening Inputs, expanding the toolbar and entering fullscreen. Normal cards and SVG overview follow the size
  continuously, with one immediate fit per animation frame. Manual pan and zoom
  remain available until the next resize, new graph or completed validation;
  **Fit** and **⌘/Ctrl+0** still fit on demand.
  A hidden canvas is not fitted at zero size. Initial observer delivery preserves
  the existing opening or restored camera, and closing the canvas cancels pending fits.
- Both Chrome's **Open Canvas in Browser** page and the VS Code sidebar fit a new
  graph after its first layout, and fit again when a current validation finishes
  with measured graph results (passed or failed). Fitting waits for the updated
  cards or SVG overview and measures the current viewport. A replaced pending
  layout reschedules the fit instead of losing it. Ordinary source updates,
  stale/refused/cancelled validation and restored native cameras retain navigation.
  If an existing Chrome tab still has an older bundle, reopen it through
  **Open Canvas in Browser** after updating the extension.
- The footer **Canvas guide and shortcuts** button (or **?**) opens connection
  meanings and navigation shortcuts. The local canvas no longer has a permanent
  connection guide or readable-zoom prompt. **Fit → Readable detail** and **Shift+2**
  zoom to the selected node, or the visible node nearest the camera centre. This
  also works in SVG overview without running the model. Typing in inputs does not
  trigger canvas shortcuts. The local minimap starts collapsed; **Overview** opens it.
  At canvas widths up to 600 px, Overview sits above the selected-node or
  connection panel. Both share one layout, so expanding the minimap cannot cover
  the node name, connection count or trace buttons. Wider canvases keep them
  side by side. This follows the canvas width, including when Inputs is open.
  Keyboard focus leaving a popover dismisses it, including when **?** opens the
  guide from another menu. Arrow keys, Home and End navigate the Canvas menu.
- Source notes open from a single footer count. Error counts retain their error label;
  the full diagnostic messages, **Locate in graph**, and source links remain available. Escape or an outside
  click closes the notes.

Module issue badges count unique captured diagnostic IDs across the module and its
descendants. The same exception propagating through callers counts once; independent
errors and source warnings stay separate. Clicking an issue badge opens its records,
including recorded shape/dtype facts, with Locate in graph and available source links.

## Module tiles

Input and output shapes stay visible on collapsed modules at every zoom tier.
Zooming in adds source links and editing controls;
it is no longer required to reveal the tensor dimensions. Shape values use larger
text than their In/Out labels. Each captured input/output keeps its own row,
including scalar, non-tensor and unknown values.

Captured-operation counts and the **Measured boundary** badge are removed from
tiles and expanded-module headers. Unchanged calls say **Input unchanged** without
an operation count. Stale, unavailable, source-only and limited evidence, plus
actionable issue badges, remain visible. Capture records and their details are
unchanged.

Captured tiles reserve height for the header and shape rows, adding
room for source links, warnings and other content when present. A normal tile with
two shape rows and a source link is 195 canvas pixels tall. Layout
and port geometry use that same height; multiple inputs or outputs grow the tile.

The tile styling uses a soft border, a small module icon well and a clear title/path
hierarchy. Dimensions have the strongest text emphasis; dtypes align to the right
in smaller, muted text. Input and output rows have subtle direction accents, with
lavender on outputs. Source links follow the shapes at inspection zoom. Header
actions brighten on hover, selection or keyboard focus. The top-right expand icon
opens the block; there is no duplicate footer button. Complete value tooltips,
warning labels, focus indicators and existing expand/edit/source actions remain.

Input, output and individual operation tiles share the module tiles' soft borders,
12px corners, icon wells and title hierarchy. Input/output dimensions are emphasized
with smaller right-aligned dtypes and complete value tooltips; outputs use the same
lavender accent. Their shape rows stay visible at overview zoom. Operation summaries
also remain visible at every zoom tier, omit routine measured labels and show only
the available input/output rows. Stale evidence, missing-measurement states and
warnings remain explicit. Inspection adds parameters and source controls, with source
links below the operation details. Existing card sizes, port positions and connections
are preserved.

## Source controls

At inspection zoom, nodes show a top-right Edit control and a clickable source
file:line where the host has a mapped source span. Edit expands that node into a
scrollable form; closing restores its original size. The floating selection button
and source-edit modal are removed. Source scope and effects remain in the form's
disclosure. Proposals still require native VS Code diff review and Apply/Discard.
Captured forms retain their original selection until the review receipt arrives,
even when Apply replaces the graph. Closing preserves a replacement graph's dimensions;
resizing the canvas keeps the active form visible. Unsupported captured edits stay disabled.

At narrow widths header controls wrap and secondary button labels collapse; accessible names
remain. Footer controls wrap without horizontal scrolling. Fullscreen keeps its controls
inside the canvas, including the menu used to exit. The smaller minimap is collapsed
by default on the local page; opening it retains its camera gestures and keyboard
navigation. Graph layout, node styling, source review and capture behavior are unchanged.


## Canvas appearance

The shared Canvas has two appearances: **Light · Paper** (warm neutral background, white cards) and **Dark · Ink** (muted sage graphite surfaces, flat cards, and a pale sage selection outline). Both use bundled Inter for UI text and JetBrains Mono for values, paths, and dtypes. Each card's line glyph takes its block family's colour. Ordinary Canvas pages do not request Google Fonts. Families, glyphs, and the legend are in [Canvas block families](canvas-block-families.md).

Click the sun/moon switch in the header to change appearance. The choice is saved. VS Code's **Set Color Theme** action also offers **tensorViz Light** and **tensorViz Dark**. With Appearance set to **Default**, the native Canvas follows the editor theme. An explicit Light or Dark choice takes precedence. Choose **Default** again to follow VS Code. The local browser keeps its own preference.

**More tools → Display settings → Appearance:**

| Choice | Behavior |
| --- | --- |
| Default | Uses the surface default |
| Light · Paper | Stays light when the host theme changes |
| Dark · Ink | Stays dark when the host theme changes |

| Surface | Default |
| --- | --- |
| Local browser, plugin browser, graph-file viewer | Paper |
| Native VS Code Canvas and the embedded MCP panel | Follows the host light or dark theme |

Appearance changes keep graph data, review labels, selection, and the camera. Added, modified, removed, and inputs-changed states keep their colors and text labels. Paper uses darker semantic colors on white, and its selection ring is charcoal. Ink uses a 1.5px pale sage outline (`#B7C5BB`) with no halo. Traced connections use sage in both themes (`#AABFB2` in Ink, `#4D6A58` in Paper). Selection stays outside a review outline, so choosing a node does not hide its change state. The editor's own blue selection indicator is not drawn on cards.

Ink surfaces are muted sage graphite: app `#1D1F1D`, canvas `#171817`, popovers `#242624`, cards `#2A2C2A`, selected cards `#323532`, borders `#3F433F`. Text is `#E3E8E2`, `#A4ADA5`, and `#98A199`. Interaction uses sage. Block family is the coloured glyph, and status keeps its icons and semantic colors. Connections use `#636B64`. The dot grid follows the camera and does not snap. SVG export names Inter and JetBrains Mono with system fallbacks rather than embedding the font files.

## Search

Search is one pattern over four fields, entered at the top left of the canvas:

| Query | Matches |
| --- | --- |
| `conv1` | node id or label |
| `type=Conv2d` | layer type name, display name, id, or traced class |
| `shape=256` | the in/out tensor text the card renders |
| `attr:padding=1` | one constructor parameter by name |

Hits inside a closed module are reported on that module. They do not open it. A nested hit rolls up to the outermost closed frame, because that is the card on screen. Choosing a result moves the camera and selects the node. A malformed pattern is reported as invalid, not as zero matches.

Inside an open module, a card subtitle drops the prefix the frame already shows: inside `stage1`, `stage1.conv1` reads `#conv1`. A top-level node keeps its full id. A node whose id is not prefixed by its parent keeps the full id.

## Overlays

`Parameters` uses the introspected parameter count, including a known zero. Counts describe each module and the parameters registered under it. Counts on nested or shared modules are not an additive model total. Rounded legacy summaries are not used.

`Output tensor bytes` reads the captured shape, dtype, and logical byte size (`numel` times element size) of a complete single-tensor output. That is the tensor's logical footprint, not allocated storage, peak memory, or a profiler sample. Views and aliases can describe the same storage more than once. Empty tensors are a known zero. Scalars use shape `[]`. A tuple, list, dictionary, or result without complete output evidence stays uncoloured. Stale nodes supply no parameter overlay. The legend scale is taken from the values on screen.

The ramp is blue, then amber, then red. Green already means "measured" on this canvas, so the overlay does not use it.

## Editor shape hints

After **Refresh graph** completes a capture, output shapes can appear after the executed expression in the Python editor, including helpers reached from `forward()`. The hints describe that captured run. Unexecuted branches and missing or stale source evidence have no hint. VS Code inlay hints must be enabled.

## Module focus

Entering a module gives its immediate children the canvas. Ancestors stay in a breadcrumb. A module without recorded children does not open an empty scene. Focus is a display projection: it does not write positions or synthetic wires into the saved graph.

The compact navigation row is the only persistent scope identity: Back, Overview, ancestor and current breadcrumbs, sibling menus, and Flow. After navigation, keyboard focus moves to the current breadcrumb. Its tooltip keeps the module type and direct-child count. There is no second title banner and no bottom module-name bar.

On first entry the canvas compares horizontal and vertical layouts and keeps the orientation that fits, staying with horizontal unless the other orientation is clearly better. Resizing does not rotate the graph. Entering, going back, changing layout, and finishing a recapture fit the destination after its shapes are committed. Selection, dragging, and pan or zoom keep the camera.

## Operation input and output shapes

Captured operation cards show an `IN` row for every tensor operand and an `OUT` row for every tensor result. Matrix multiplication shows both matrices and any bias. Reshape, transpose, squeeze, unsqueeze, and clone keep their tensor input. Repeated uses of the same tensor stay separate inputs. An unavailable shape is labelled **Not recorded**. Scalar options such as `dim` stay off the card; **View captured details** still has the full arguments. Failed operations keep their recorded inputs next to the failure.

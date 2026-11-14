# Changelog

## 0.1.45

- Add native compiled runtime packaging for Intel Mac, Linux ARM64 and Windows x64/ARM64, with architecture-specific verified installers.
- Separate core service startup from optional PyTorch model capture dependencies and improve unsupported capture-environment diagnostics.
- Preserve Linux x64 glibc 2.31 support, version alignment and tested upgrade/rollback gates.

## Contents

- [Unreleased](#unreleased)
- [0.1.45](#0145)
- [0.1.44](#0144)
- [0.1.43](#0143)
- [0.1.42](#0142)
- [0.1.41](#0141)
- [0.1.36](#0136)
- [0.1.35](#0135)
- [0.1.34](#0134)
- [0.1.33](#0133)
- [0.1.32](#0132)
- [0.1.31](#0131)
- [0.1.30](#0130)
- [0.1.29](#0129)
- [0.1.28](#0128)
- [0.1.27](#0127)
- [0.1.26](#0126)
- [0.1.25](#0125)
- [0.1.24](#0124)
- [0.1.23](#0123)
- [0.1.22](#0122)
- [0.1.21](#0121)
- [0.1.20](#0120)
- [0.1.19](#0119)
- [0.1.18](#0118)
- [0.1.17](#0117)
- [0.1.14](#0114)
- [0.1.13](#0113)
- [0.1.12](#0112)
- [0.1.11](#0111)
- [0.1.10](#0110)
- [0.1.9](#019)
- [0.1.8](#018)
- [0.1.7](#017)
- [0.1.6](#016)
- [0.1.5](#015)
- [0.1.4](#014)
- [0.1.3](#013)
- [0.1.2](#012)
- [0.1.1](#011)
- [0.1.0](#010)
- [0.0.3](#003)
- [0.0.2](#002)
- [0.0.1](#001)

Extension **0.1.44**. The plugin host in this repository is **0.1.44**.

This tree is the visualization product: Open as Graph, graph diff, execution
cache (`off`, `shadow`, and `on`), code focus, graph export, MCP, Hugging Face open, source
proposals, JAX/Flax capture, and website sign-in and share. Latency profiling,
training-step capture, product analytics, and the hosted website application are not in this tree.

## Unreleased

## 0.1.44

- Build and validate Linux x64 native packages against glibc 2.31, retaining fail-closed checks for older or unsupported systems.
- Update Python and JavaScript dependencies to address audited advisories and add Node.js 22 JavaScript dependency checks.
- Document explicit Codex plugin reinstallation after marketplace refresh, with legacy-cache regression coverage.

## 0.1.43

- Restore mixed module training modes after inference capture, including custom training-mode side effects, shared submodules, and failure cleanup.

- Harden graph opener allocation and retired-session handling so repeated graph opens do not exhaust capacity or replay stale updates.

- Show searchable model and supported effort choices beside the chat composer. Add a model/effort split menu to **Explain this view**, with separate saved preferences and the actual execution choice visible during and after each pass.

- Keep graph cards compact at every zoom, wrap full technical names, and move ordinary parameter lists into selection details. Size Input/Output cards from actual rows instead of leaving empty bodies; keep SVG and connection geometry aligned.

- Resolve selected-node signatures and author docstrings from source without an agent. Prefer source documentation over standard-layer reference text, preserve full details and citations, and keep ambiguous or dynamic targets unresolved.

- Explain selected nodes with compact source facts, standard-layer descriptions and optional source-grounded agent interpretations. Keep exact callables, execution evidence and linked sources inspectable; invalidate explanations when source, capture or agent settings change.

## 0.1.42

- Prepare the workflow cutover with a new reviewed extension version above the legacy 0.1.41 release. The companion plugin host is 0.3.110, above the public 0.3.109 release. Existing published versions and artifacts remain unchanged; publication still requires the release gates and verification checklist.

## 0.1.41

- Release the visualization product directly from tensorviz-org/tensorviz, with reviewed source versions and identical VSIX bytes on Marketplace, GitHub, and Open VSX.

- Show runtime download totals when available, update transferred bytes during smaller downloads, and use an accurate setup progress title. Keep cancellation and download verification unchanged.

- Refresh plugin discovery expectations and the generated tool reference after the new setup and private Canvas tools; replace obsolete Get Started tour copy.

- Share bounded captured model facts through `get_model_context` and native/ACP Canvas chat, with source freshness checks and explicit omissions. See [compact model context](../docs/model-context.md).

- Compact chat permissions into a shield icon and themed keyboard-accessible menu. Move active generation to animated dots below the latest message, remove the idle header circle, and shorten settings with an explicit connection-recheck result. Keep unrelated general-assistant work outside the shared research/project prompt scope while supporting relevant mathematics.

- Preserve the actual model-import exception, cause chain, traceback and observed runtime for same-task diagnosis. Add bounded project-file and machine inspection plus exact host-reviewed command jobs; every setup mode still asks before machine commands, and provider-native direct shell stays disabled. Show task activity with status icons and disclose retained failure evidence in chat.

- Generate setup permission titles, explanations and both button labels with the active agent from structured host operation facts. Share one typed presentation tool and prompt section across operations/providers, preserve exact plans in Details, and resume the same task when an explanation is pending. Approval still follows the selected host policy.

- Publish verified Python completion before input discovery and continue the same agent task into missing inputs and capture. Share typed setup progress between UI and tools; centralize provider prompts in a typed section registry.

- Add a visible chat permission selector: **Ask every time**, **Auto setup**, and **Full setup access**. Automatic approvals leave receipts; Auto setup still asks before source builds or changing existing Python. New chats return to Ask.

- Review agent-driven environment creation, source builds and exact dependency installs inside Canvas chat. Show **Needs approval**, expandable action details and explicit Allow/Deny; cancel pending requests when the task stops or context changes. Manual setup retains native reviews.
- Unify Python and model-input setup with shared progress, concise agent/manual choices and one **Complete setup** reminder when the sidebar is closed. Resume the unfinished step with saved choices and draft edits intact.

- Lead Python setup with **Set up project with agent**, explaining why the model needs Python. Keep manual recovery available, remove repeated diagnostics, and give the provided Python a readable name. Agents can prepare the selected opening’s environment through the same reviewed installation flow.

- Use the tensorViz panel as the single starting screen. Remove the separate first-run welcome guide; **Get Started** reveals the panel without starting runtime setup.

- Sync the eligible extension source through 2026-10-30, preserving this repository's visualization scope and release identity.
- Enable automatic validation after relevant saves and execution caching by default, with opt-outs and source/environment freshness checks.
- Share class-scoped input declarations across setup, capture and generated source; avoid disabled statistics hooks on ScriptModules.
- Add bounded source/computation queries, nested declaration review, multi-file agent drafts, checkpoint/output evidence, explicit code focus and streaming Canvas chat with optional ACP sessions. Inline editing forms are replaced by agent-assisted native source review.

## 0.1.36

- Share compact canonical model context across MCP and native/ACP Canvas agents: module aliases, complete retained call lists, registered parameter counts, ordered tensor shapes and explicit unknowns. Reuse matching measurements, refuse stale evidence and reduce the broader graph preload when compact facts are available. Inspection never runs the model or changes review permissions.

## 0.1.35

- Automatically highlight the outermost matching layer when clicking source code. Remove the instance chooser and internal node IDs; specific layer and standalone operation lines retain direct focus.

- Attach current graph, shape and Canvas context before agent inference, and reuse complete verified source and pending drafts without redundant reads. Large graphs retain explicit limits and continuation. Progress now ends preparation and tool activity when those steps finish.

- Add opt-in local agent test recording through `TENSORVIZ_AGENT_RECORDING_DIR`: timestamped prompts, replies, tool outcomes and review/execution state, with explicit recording limits. Disabled by default; no upload or permission change.

- Preserve ACP reply message boundaries so progress preambles and final answers appear as separate paragraphs. Ordinary streamed chunks remain intact.

- Find exact computation anchors from recorded module, operation and state names. Bounded lookup preserves repeated candidates and aliases, exposes original capture records, and keeps historical or missing evidence explicit.

- Clarify that candidate Run uses the retained capture setup while Apply uses the current saved model setup. Explicit validation checkpoints and input overrides are not automatically saved to that profile.

- Use one explicit-file proposal path for Canvas Python changes, including entry files and imported configuration. Every replacement names its file and digest; all diffs share one review decision and undoable edit. The complete review inventory is bounded to 256 files and 8,000,000 bytes, including one-file changes; oversized inventories are refused without truncation. See [source editing](../docs/source-editing.md#source-positions-and-review-limits).

- Reuse bounded source indexes across agent queries. Source bytes and freshness are still checked on every request; edits rebuild cross-file relationships while unchanged files retain their parsed structure.

- Show checkpoint-load results in reviewed candidates, including failures before a forward pass. Native Inspect, browser pages and MCP history retain bounded metadata tied to the reviewed source and checkpoint bytes; successful loading remains separate from forward correctness.

- Let Canvas agents inspect graph-recorded local Python dependencies with `read_model_source(file)`, so imported blocks can be explained from their implementation. Reads stay bounded and verify current graph digests.

- Reuse native Codex and Claude connections across Canvas turns and prepare the selected agent while composing. Reconnects replay retained history and evidence; preparation cannot use tools or bypass execution review. Five idle minutes release the process.

- Retain bounded graph evidence in the Canvas conversation across Stop, agent reconnects, idle expiry and effort changes. Restored evidence keeps its original revision; New chat clears it. Continuing ACP sessions do not replay the evidence or add model calls.

- Keep unobserved input shapes unknown in agent graph inspection, matching the existing output-shape boundary. Cached placeholders on unreached or unsupported capture nodes can no longer appear as measured input evidence.

- Default Canvas chat to medium effort for Codex and Claude models that support it, including preparation and both native and ACP requests. Explicit saved effort choices are preserved.

- Start ACP preparation when the selected agent is ready, reuse recent Codex routing checks, and skip Codex preparation when Send is already waiting. Supported Codex adapters use a fixed session title to avoid an unused model request. Claude keeps its existing early-Send sequence after comparison showed a regression; answer models, effort and review gates stay unchanged.

- Reduce Canvas chat work after Send by attaching verified selected-layer facts. The opt-in ACP path connects and prepares context while composing, with bounded cancellation and no preparation tool access. Preparation can consume agent usage before Send; model/effort and native review remain unchanged.

## 0.1.34

- Moving the editor cursor onto a layer focuses that layer again, instead of
  offering an internal declaration ID beside it. Model files that contain a form
  feed and non-ASCII text on a later line open again.
- Git review: moving the cursor onto a line with one change focuses it, even when
  unchanged layers share the line or the change is used by several captured calls,
  and selecting the second call of a change selects that call, so Next moves on.
  Changes in models under nested project folders join their declarations and the
  Canvas's captured graph, in VS Code and in the standalone plugin. Declarations
  and layers assigned inside `try`, `with` and `match` blocks are reviewed, and a
  later write there keeps an old capture from lending its context to a changed
  layer. A layer assigned by different calls in different branches is listed once.
- On lines with non-ASCII text, such as a `d_μ` argument or an emoji in a string,
  clicking a Canvas card puts the cursor on the right character, Git review and
  refused edits select exactly the code they name, and a Canvas parameter edit of
  an `nn.Sequential` child changes that child instead of the `nn.Sequential(...)`
  around it. Graph edits no longer land one line low in a file with a form feed
  or U+2028 above the class, and no longer leave half an emoji on a last line
  without a final newline. JAX shape hints appear on lines with non-ASCII text.
- Execution cache: a run that loads native code other than the interpreter's,
  torch's or numpy's own, or lists a folder or checks for a file outside its
  tracked sources, is not stored, and a stored result is not reused after an
  installed Python file it read or loaded before the run started changes, for
  example most in-place edits to torch's Python files or a reinstall of the same
  version. Files
  linked into the Python installation's folders, such as Debian's `sitecustomize`
  or Homebrew packages, count as installed; code loaded from outside them, such
  as a `PYTHONPATH` folder, means the run is not stored. On macOS and Linux, a
  pre-started worker whose pre-loaded files changed while it waited starts fresh,
  so that run uses the edited code. A folder can set
  `tensorviz.executionCache.mode` to `off` or `shadow` for its own runs without
  clearing other folders' results; `off` for the whole window still turns the
  cache off everywhere. A workspace inside the Python installation's folder is no
  longer treated as installed files.
  Reused results say "model not run again": Python still runs briefly to check that
  nothing changed.
- After a save-triggered run, Run model no longer shows "running" while its review
  waits, and dismissing that review shows the earlier result again. Change inputs
  and setup saves after it run at once, and an unchanged save no longer restarts
  an accepted change.
- Canvas agent chat sends its background context preparation at most once per
  conversation; changing model or effort, Refresh and reopening chat no longer send
  another. Use Agent during an agent check no longer leaves the agent stuck at
  checking, and a stalled warm-up says the connection timed out.
- In a file with several model classes, graph setup uses a `# tensorviz: input=`
  comment only for the class it sits directly above or inside; one outside every
  class applies to none of them, and setup asks for inputs. New models from
  templates put it directly above the model class; move it there in template files
  generated earlier with helper modules. The source-only drawing, a trace without
  saved inputs and the project graph follow the same rule, so a class is never
  drawn with another class's declaration, and
  declaration text inside a string does not count. A trace without saved inputs
  reads only the model's own file, so it can still give a class's declaration to
  the file's other model when only another file shows that the class is a model.
  Model files with carriage-return line breaks are read the same way. A trace also
  reads a file saved as UTF-8 with a byte order mark this way, and its
  `# tensorviz: out_shape=` block shapes are read again. A
  class whose declaration is an expression, such as `torch.randn(4, 16)`, is drawn
  from its source until it is traced; in VS Code a class without a declaration of
  its own is drawn with the input shape an earlier build saved for it, if any.
- The main Canvas trace worker is disposed with the extension, so its warm Python
  process exits.

## 0.1.33

- Release the changes merged through `722b8363eab0`.

## 0.1.32

- Release the changes merged through `b5787bbcb258`.

## 0.1.31

- Keep Canvas coding-agent connections open across replies and minimizing; prepare
  context when chat opens. Opening chat can consume provider usage before Send.
- Default agent effort to medium when supported, preserving explicit choices.
- Attach current model/selection facts to requests and display first text immediately.

## 0.1.30

- Saving changed model code or a known Python dependency now automatically
  validates the selected model after a 1.5-second pause, using its saved inputs,
  device and environment. Disable Trace On Save to opt out. Source-only inspection
  stays nonexecuting, and Accept/Apply still runs once.

- Accepting a model source proposal saves the reviewed file and automatically runs
  the model with its saved inputs and device. Canvas Accept and native Apply share
  this flow; save failures and intervening edits stop execution with an explanation.
- Graph review opens the first located change's module. Verified constructor
  locations identify deep edits, including proven Sequential replacements.
  Next/Previous visits each affected call once; ambiguous or unobserved calls
  remain unlocated.

## 0.1.29

- Turn on reuse of identical runs by default (`tensorviz.executionCache.mode` is now `on`). When you Run again, undo an edit, or reopen a model with exactly the same saved sources, setup, inputs, seed, environment and tensorViz backend as an earlier Run in this window, tensorViz shows that earlier graph without starting Python, labelled `Reused identical result from <time> (no Python ran)`. **Run fresh** runs it anyway; set the mode to `off` to always run.

## 0.1.28

- Remove the Canvas heat-map overlays (**Parameters**, **Output tensor bytes**) and the card colours they painted.

- While a model runs, **Cancel** is an icon beside **Running…** instead of overlapping it.

- Selecting an Input or Output card shows its normal outline instead of a blue box with handles.

- **Publish** in the toolbar adds the graph to your tensorViz account (it was Add to tensorViz in the model menu). The model menu no longer links to the graph file viewer.

- The header has no expand/collapse toggle: controls are labelled when they fit and icon-only otherwise, and the environment and saved setup are in the model menu.

## 0.1.27

- Release the changes merged through `08ea99c88430`.

## 0.1.26

- Release the changes merged through `0f45faa01f70`.

## 0.1.25

- Release the changes merged through `c5565c67760d`.

## 0.1.24

- Make **Code focus** a one-click toggle at the start of the Canvas header. On, selecting a node opens its source beside the Canvas without taking keyboard focus, and double-clicking or pressing Enter on a layer moves focus to it. Off (the default), exploring the graph never opens or dims code. **Dim unrelated code**, scope and opacity move to the chevron beside it. Source links and the footer **Source** button open code either way, and each Canvas surface remembers its choice.

- Remove the **Edit** control from Canvas layer and module cards, along with its inline source form, which did not work reliably. Source changes proposed by a connected coding agent still open in VS Code's native diff.

- When the Canvas fills the window (its side bar maximized), showing source through Code focus, a source link or **Source** first restores the split, so the code appears beside the Canvas instead of behind it.

- Fix double-clicking a card on the Canvas, which only selected it: a module now opens and, with **Code focus** on, a layer's source opens with keyboard focus, as Enter does. Double-clicking empty canvas still does nothing.

- Keep a captured graph current when VS Code reloads an unchanged model file. The opening no longer reports "Source changed" and drops its capture, and **Use Agent** no longer refuses setup, when only the editor's document version restarted.

## 0.1.23

- Show captured dictionary keys and keyword input names on Input/Output cards, module boundaries and port tooltips. Older graph files recover names from their retained capture; unnamed values use readable one-based labels.

## 0.1.22

- Release the changes merged through `d28e99d779fe`.

## 0.1.21

- Release the changes merged through `e1d1a830011d`.

## 0.1.20

## 0.1.19

- Show captured dictionary keys and keyword input names on Input/Output cards, module boundaries and port tooltips. Older graph files recover names from their retained capture; unnamed values use readable one-based labels.

- Give the graph more room: device/host details now share the model toolbar, and focused modules use one compact breadcrumb and Flow row instead of a second title banner. Back, sibling switching and the full run configuration remain available.

- Add `on` to the preview setting `tensorviz.executionCache.mode` (default still `off`). When a Run has exactly the same saved sources, setup, inputs, seed, environment and tensorViz backend as an earlier Run in this window, tensorViz shows that earlier graph without starting Python, labelled `Reused identical result from <time> (no Python ran)`. **Run fresh** runs it anyway, and **tensorViz: Clear Execution Cache** drops the stored results. Agent (MCP) runs always execute.

- Serve a reused execution-cache result with the documents' current versions, so code focus and source review stay aligned with the editor after the original capture.

- Refresh the Canvas graph: softer graphite surfaces with a faint sage tint in the dark theme, a pale sage selection outline and trace in place of tldraw's saturated blue ring and the cyan accent, and card glyphs coloured by block family (convolution, normalization, activation, pooling and more), resolved from each block's actual class or operator. Your own modules stay neutral and show their recorded composition in the inspector; a **Blocks** legend lists the families on screen. Connections are thinner and neutral, multi-tensor rows read `IN 0`, `IN 1`, `OUT`, and collapsed modules no longer reserve empty space.

- Record FakeTensor captures in the canonical semantic contract. Shape, dtype, device and strides are kept, tensor memory is not read, and `valuesUnavailable` is `fake_tensor`. Meta tensors are not fingerprinted either. Attention modules trace on their decomposed path, and that switch does not rewrite existing error frames.

## 0.1.18

- Keep the ranked Python list visible when the recommendation is already saved, after an explicit choice, and from the header’s **Change environment** action. Creation stays below the ranked list.

- Check every detected Python environment against the model's own libraries when opening a graph, not only against tensorViz's capture libraries. An environment that can already run the model is used straight away. This applies when nothing is saved, or in place of a choice tensorViz made itself. Choices you made are never switched. Otherwise the environment panel opens with the environment needing the fewest changes recommended and preselected, each environment showing what it still needs, a search field, and **Create new environment**. Nothing is installed without your approval.

- Stop drawing static-analysis notes, such as "transformers is external to the workspace", inside graph blocks, where long notes were clipped. The footer's source notes list still shows every note. Warnings and errors still mark their block with the issue badge.

- Close the execution details when you choose Run. Progress and Cancel stay in the toolbar, and the device strip reopens the details. Run model and Run again no longer open them for a run that starts at once.

- Run the model as soon as it is opened. Open as Graph, the model lists, an agent's open_graph, Retry and saving setup now capture the saved call once on its saved device, with no Run model or Run click. The execution strip still shows progress, Cancel and Run again. On a captured graph, **Run model** still opens the review for another device. The editor's optional auto-open still waits for Run, and saves still refresh source without running.

- Stop hardware readings without signalling other processes. Stopping them before a missing tool such as nvidia-smi had reported could send a stop signal to the whole extension host process group.

- Add a preview setting, `tensorviz.executionCache.mode` (default `off`). In `shadow` mode every Run still executes your model; tensorViz also checks whether an identical earlier run could have been reused and writes the result of that check to the tensorViz output channel. Nothing you see changes.

## 0.1.17

- Keep Code focus and node navigation working after model setup refreshes in Cursor and VS Code. Setup updates no longer replace the current graph revision with an older one for the same source.

## 0.1.14

- Replace Export graph in the VS Code Canvas model menu with **Add to tensorViz**. After a one-time Google sign-in through tensorviz.dev, one click sends the graph to your tensorViz account through the website's upload API, as private, and opens its Share dialog, where you choose who can see it and copy the link. **Sign in to tensorViz** and **Sign out of tensorViz** manage the sign-in, kept in VS Code Secret Storage. Open Canvas in Browser keeps Export graph.

- Make agent-guided and manual model setup explicit alternatives. Attach Recommended to the agent action; show fields and Open graph only after choosing manual. Preserve unfinished edits when switching routes, and align the panel title, button icons and source links.

- Complete the sidebar home state before an asynchronous first model opening mounts its iframe, preventing a first-use crash.

- Show captured device and call counts in the execution details without adding new metrics.

- Replace technical example-call setup and factory/input checkboxes with explicit model creation and test-data dropdowns. Lead with recommended agent help and the selected Python; preserve manual setup, method edits and nested bindings. Link discovered parameters/config fields to verified source declarations. Share the edited draft and evidence with the agent, preserve setup follow-up context, and hand off to execution review without running the model.

- Show readable module paths, total matches and paginated results in Canvas search; reach every layer with keyboard selection and retain typed-query grouping.

- Correct model setup and Canvas guide copy: saving continues to execution review; Run model reviews the device and optional timing before execution.

- Keep the Canvas footer Fit label inside its full-size button, including narrow sidebars.

- Add a shared project-level Claude/Codex feature workflow with scoped plans, CSV progress sheets, generated checklists and HTML evidence reviews. Maintain scoped plans, checklists and before/after verification for the UI audit repairs.
- Replace the separate capture button with one labelled Run model action, then Run.
- Keep execution details scrollable with a visible scrollbar and keyboard access; contain panel gestures while keeping the header and action buttons visible above Canvas navigation and floating chat.

- Identify the Python worker host and saved profile; expose input shapes and inference evidence with Edit links to exact TOML fields. Saving retires a waiting review so updated inputs receive a fresh review.
- Offer Run again in settled execution details, requesting a fresh review before execution.
- Simplify the compact execution strip to device, machine and disclosure arrow, removing run-status labels while retaining progress and outcomes in the details.
- Open execution details only from an explicit control and keep them closed through resource updates, reconnects and preparation/capture transitions.
- Refresh source on save without running the model. The legacy capture-on-save setting no longer starts computation in sidebar Canvas.

## 0.1.13

- Open plain JAX functions and Flax Linen/NNX models through the existing saved-input, Python-environment and Canvas workflows. JAX shapes are inferred from a traced program; runtime timing and PyTorch source-edit recipes remain unavailable for these captures.
- Preserve large model graphs through rendering, transport and graph-file export, with explicit capture limits and truthful partial-coverage warnings.
- Improve graph-detail and review selectors with keyboard navigation, group Canvas actions, show readable search names, and make input forms and validation details easier to scan in Paper and Ink themes.
- Keep reviewed setup and capture retries usable after a completed opening.

## 0.1.12

- Offer **Try a starter model** on the empty Canvas home and in the walkthrough; creation uses the existing template picker and does not capture or install dependencies automatically.
- Link examples and issue forms from the listing. Source for the extension is this repository.

## 0.1.11

- Make Browse select Python before checking or installing its libraries; return missing-library setup to Canvas's visible progress and recovery flow, and preserve the previous choice on cancellation.
- Keep selected Python, library setup status/progress and Add libraries here visible while scrolling; make new environments optional, retain package failures with explicit retry/source inspection, and continue automatically only after library verification.
- Simplify environment setup into one panel with an editable venv name, named progress and details on demand; offer source inspection after failure and check unavailable distributions before asking to build them.
- Keep failed model environment creation visibly failed, including the unavailable package, instead of returning to normal setup and repeating the same build.
- Prioritize a model's local imports within the existing source limits so large repositories retain base classes; Kimi K3 now has a partial source preview with unresolved coverage still visible.
- Show source-preview failures with Retry graph and Set up model instead of leaving a blank Canvas beneath the toolbar.
- Start Create model environment on the first click after cancelling setup or stopping an agent, retaining the selected model and showing fresh creation progress.
- Create a separate model environment from Canvas, review its required libraries and pinned Git source builds, show setup progress and cancellation, then select verified Python and continue the same model opening.
- Restore Cmd/Ctrl+C for selected native Canvas chat text and add an explicit Copy action to assistant replies.
- Explain already-declared model dependencies the installer cannot handle; distinguish capture libraries from model readiness and offer agent help and source inspection instead of repeatedly checking the same Python.
- Restore Cmd/Ctrl+V in native Canvas chat through VS Code's clipboard; preserve multiline drafts, cursor selection and the prompt limit without sending automatically.
- Check only tensorViz's own runtime on the welcome screen; defer project Python and capture-library checks until a model is selected.
- Keep one starting screen with Open as Graph and Open PR through service setup and connection, showing progress and recovery inline.
- Check runtime readiness automatically in trusted windows without installing dependencies or executing model code; keep failed model openings available to retry.
- Reuse a compatible VS Code Python selection or an unambiguous project environment when opening a model, while preserving broken saved choices and explicit recovery for ambiguous environments.

Model environment creation installs into a separate venv; it cannot make packages
available for an unsupported Python or operating system. Source inspection remains
available without running the model, and partial previews retain their coverage
diagnostics. Installed acceptance and compatibility limits are tracked in the
[setup and compatibility guide](https://tensorviz.dev/install.html#compatibility).

## 0.1.10

- Use Agent replaces Get agent help on setup, failed-opening and environment-recovery screens: your installed, signed-in coding agent completes the setup and continues until the graph is captured, with progress and Stop in the Canvas chat.
- Use Agent keeps your saved setup values, fills missing example inputs and discovered constructor objects through generated factories, and reports anything it cannot complete instead of claiming success.
- A setup draft the form cannot edit, such as an agent's rejected attempt, no longer leaves the Canvas blank; the graph and chat keep working.
- Any other unexpected Canvas error shows what failed, with details for reporting it and a Reload Canvas action, instead of a blank view.

## 0.1.9

- Ask your installed, signed-in Codex or Claude Code agent about the open model from a minimizable Canvas chat, with agent, model and effort settings and readiness checks.
- Preview an agent's proposed source change as a draft graph beside its code diff, then Accept, Discard or Continue editing; Enter revises the draft without applying it.
- Keep added layers and nested module replacements visible after Accept, marked Not run, until the graph is refreshed.
- Arrange the graph Auto, Horizontal or Vertical from the Canvas toolbar.
- Keep tensorViz available in the secondary sidebar before opening a graph, with discovered models and a starting screen that needs no runtime setup.
- Offer Open as Graph when the active Python file contains a discovered model, with a setting to disable these notifications.
- Show a searchable workspace model picker when Open as Graph has no selected Python file, with class, file and folder identities and a file-browse fallback.
- Redesign starting, empty, connection and recovery screens with theme-aware layouts and reduced-motion support; remove animation playback buttons.
- Group active graph and pull-request review controls so titles, actions, change navigation and coverage remain usable in narrow sidebars.
- Separate Model and Example inputs in setup, keep Open graph visible while fields scroll, and group execution options under Run settings.
- Fix setup crashes when switching saved Python input functions or replacing discovery with different configuration metadata.
- Load drawing fonts, icons and translations from packaged assets in production Canvas.
- Simplify runtime installation consent, remove the duplicate Cancel action, and show download failures with useful, redacted diagnostics and installed-user recovery guidance.
- Retry temporary runtime download failures within a bounded, cancellable limit and record local provisioning stage timings.
- Keep numbers in English PyTorch error explanations consistently formatted across system locales.
## 0.1.8

- Open a Hugging Face model or safetensors link to browse its stored tensor shapes, read from the file header without downloading the weights.
- Capture a single-file Transformers checkpoint from Hugging Face with editable example inputs, saved as a local model folder and profile.
- Dim code outside the selected Canvas nodes by class, function or expression, including a parameter's declaration and uses.
- Open and focus captured layers that are defined in other workspace files.

## 0.1.7

- Open a nested module in its own focused view, with breadcrumbs, sibling switching and automatic fitting.
- Show environment blockers above the list, and keep the saved choice separate from whether the model can run.
- Try a model with its installed library versions when only the project's pins conflict.
- Install a project's pinned versions in its own environment after reviewing every change, with a separate PyTorch/CUDA confirmation.
- Fit the graph diff controls into two rows.

## 0.1.6

- Browse discovered models by source folder and Python file instead of one flat list.
- Show model counts on collapsed groups and retain expansion after refreshing the index.
- Keep models from separate workspace folders distinct, opening each class from its owning folder.

## 0.1.5

- Capture models in Python 3.10 environments while keeping the tool service and installer separate.
- Expose custom factories in the main Inputs form so existing multi-file models and pipeline components can keep their structure.
- Recognize verified hashes from older pip versions and keep nested Python 3.10 compatibility copies runnable.
- Publish the scoped Python and OS compatibility results, including unverified platforms and experimental Python 3.15 limits.

## 0.1.4

- Switch Canvas between Light · Paper and Dark · Ink directly from the header, including the minimized toolbar.
- Offer tensorViz Light alongside tensorViz Dark in VS Code's Set Color Theme picker.
- Follow the selected VS Code theme while Canvas Appearance is Default, preserving explicit Canvas choices and the graph camera.

## 0.1.3

- Discover existing project and workspace Python environments when a graph first needs them.
- Keep graph setup open through environment selection, dependency review and repair.
- Separate the TensorViz service Python from the environment that executes the model.
- Review missing capture and model libraries before additive installation, preserving existing Torch/CUDA builds.
- Make environment choices easier to compare with folder identities, version columns and details on demand.
- Recover plugin graph openings through the shared environment setup and packaged Canvas.
- Honor timm ViT patch embedding input dimensions during graph example discovery.

## 0.1.2

- Add Paper and Ink Canvas appearances with independent browser and host preferences.
- Use bundled Inter and JetBrains Mono fonts, compact cards and lean controls.
- Match minimap and SVG export styling to the selected appearance.
- Keep dots visible in overview views, with softer contrast in dark mode.
- Preserve the current camera and selection when changing appearance.

## 0.1.1

- Keep opening and ordinary validation lightweight.

## 0.1.0

- Share contracts, graph-host services and Canvas implementation with the tensorViz plugin.
- Separate extension packaging and test ownership from the plugin and hosted website.

## 0.0.3

- Arrange graphs using actual card sizes, aligned inputs and outputs, and consistent spacing.
- Keep connection ports stable and route residual paths clear of nested module controls.
- Preserve manual placements and settle initial capture layout after temporary status badges clear.
- Match SVG export connections to the interactive canvas.

## 0.0.2

- Show local Git changes in the graph through `open_graph_diff`, including staged
  and unsaved edits, without executing the model.
- Install the TensorViz plugin for Codex or Claude Code through approved MCP setup,
  with tool availability checks and live connection status.
- Open the current graph beside Codex chat through a single-use browser link.
- Continue graph setup after source changes interrupt a capture request.
- Link to the public Model Explorer and remove private-repository build instructions
  from the Marketplace introduction.

## 0.0.1

- Explore PyTorch models in a persistent sidebar Canvas or local browser page.
- Capture tensor shapes with saved inputs and trace model errors on the graph.
- Review source proposals in native VS Code diffs before applying them.
- Compare Git revisions and GitHub pull requests without executing their code.
- Connect an existing coding agent through MCP for setup, inspection and capture.
- Export portable graph files with walkthroughs and read-only architecture comparison.




# Changelog

## Contents

- [Unreleased](#unreleased)
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

Extension **0.1.18**. The plugin host in this repository is **0.3.20**.

This tree is the visualization product: Open as Graph, graph diff, execution
cache (`off`, `shadow`, and `on`), code focus, graph export, MCP, Hugging Face open, source
proposals, JAX/Flax capture, and website sign-in and share. Latency profiling,
training-step capture, product analytics, the hosted website application, and
release publication are not in this tree.

## Unreleased

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

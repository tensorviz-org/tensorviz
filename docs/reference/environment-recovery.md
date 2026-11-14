# Guided environment recovery — reference

<a id="guided-environment-recovery"></a>

For the short walkthrough, see [the user guide](../environment-recovery.md). This page keeps the detailed behavior and implementation notes.

## Contents

- [Creating a model environment in VS Code](#creating-a-model-environment-in-vs-code)
- [Agent diagnosis and reviewed commands](#agent-diagnosis-and-reviewed-commands)
- [Plugin interpreter roles](#plugin-interpreter-roles)
- [Canvas recovery](#canvas-recovery)
- [Shared opening recovery state](#shared-opening-recovery-state)
- [Ranked environments at opening](#ranked-environments-at-opening)
- [Detected environments](#detected-environments)
- [Extension service and project Python](#extension-service-and-project-python)
- [Interpreter roles in the shared installer](#interpreter-roles-in-the-shared-installer)
- [Capture requirement profile](#capture-requirement-profile)
- [Direct MCP help for failures](#direct-mcp-help-for-failures)
- [Environment selection redesign](#environment-selection-redesign)
  - [Automatic readiness on the extension home](#automatic-readiness-on-the-extension-home)

When a model needs Python setup, native Canvas leads with **Set up project with
agent** and explains that Python runs the model to record tensor shapes. The agent
uses the retained project and model context to prepare an environment and test
inputs, then continues the same opening. Active agent turns review environment
creation, source builds and exact package installation in chat. Manual flows keep
the native VS Code reviews. **Set up manually** reveals the ranked environments, library repair and
environment creation controls. **Setup options** returns to the agent route while
preserving local selections and the creation name; another opening starts with the
agent route. Discovery does not import the model.

The first screen has one short purpose sentence and one primary action. A single
**Learn more** disclosure contains the longer agent explanation and current setup
details. Failures, stopped setup and reconnecting messages stay visible. Manual diagnostic details add information instead of repeating
the visible summary. Managed fallback rows are labelled **TensorViz Python**, with
the complete path under **Interpreter details**; names wrap at sidebar widths.
Standalone plugin pages keep their existing manual controls and MCP help because
they use the surrounding coding agent rather than the integrated Canvas chat.

The native Canvas can review all known missing model and capture libraries in an explicitly selected empty project Python, including PyTorch and pipless venvs. It resolves one additive plan, shows the exact PyPI wheel versions in the approval details, and installs only after approval. The inline message calls out PyTorch's download size and directs models needing a specific Torch/CUDA build to a compatible environment. Hosts without that exact-wheel review keep the explicit Torch environment/build choice.

Folder-scoped selection and direct MCP failure help use the same opening.

## Creating a model environment in VS Code

Choose **Set up manually** to use these controls. When a model is blocked by missing/incompatible libraries, expand **Change
environment** for optional **Create model environment…**. Known missing-library
repair in the saved Python leads with **Add libraries here…** in the fixed footer.
Creation is also available when no existing Python is detected. It uses tensorViz's service Python to create a new venv under extension storage's
`model-environments/`, without changing the service runtime or your current
project environment. You do not need a coding agent for this action.

Edit **Environment name** before creating it (default: `<class>-env`). The name is
the actual venv directory inside its separately owned `model-*` storage folder,
not a project directory or interpreter path. Names use 1–48 ASCII letters, numbers,
hyphens or underscores, start with a letter/number and exclude reserved Windows
device names. Older clients without a name retain the `environment` directory.
Setup is one panel: technical diagnostics, creation details and existing Python
choices are disclosed on demand instead of a Choose/Confirm/Prepare rail.

Canvas shows creation, dependency resolution/downloads, review, installation and
checks. If a required project declaration is a public HTTPS Git URL pinned to a
full commit, VS Code asks before executing its build code. Other source builds
also require that review. VS Code then lists the exact resolved versions before
**Install and continue**. The plan freezes wheel hashes and project declarations;
changed requirements invalidate it. PyTorch downloads may be large. These are
stage updates, not a byte-percentage download meter.
Progress names the venv being created, not the previously selected fallback. The
host retains that name through reconnects and later model-input setup. If the
binary plan names an unavailable wheel, the installer first checks index metadata
for any distribution compatible with this Python/platform. No indexed distribution
means a named failure before offering a futile **Build libraries** prompt. An
available source distribution still needs build approval; the metadata check does
not prove that the requested version will resolve or build successfully.
PyTorch never reaches that prompt: PyPI publishes no PyTorch source distribution.
When the requested PyTorch has no wheel for the target environment (for example
PyTorch newer than 2.2.2 on an Intel Mac), planning fails at once with
`PACKAGE_UNAVAILABLE`. The message names the target's operating system,
processor and Python, plus the newest PyTorch the index offers there. Select an
environment that already has a compatible PyTorch.
While environment recovery is visible, the minimized chat bubble is hidden so
it cannot cover Cancel; the panel's agent help action remains available.

After the capture worker and model import pass, the new Python is saved for this
model's owning workspace folder and the same opening resumes. Saved execution
settings are reused; missing constructor/input settings continue to Prepare.
Cancelling or failing before selection removes only the new incomplete venv.
Creation failure remains a failed opening with its sanitized unavailable-package
diagnostic, rather than returning as a successful Prepare step. There is no
automatic retry; the same request replay cannot start another build.
The manual failure view's default action is **Inspect source only**, which executes no model
and supplies no new measurements. **Retry creation** is explicit; editing the name
does not fix platform compatibility. Expand **Change environment** to select a
different compatible interpreter. Exact diagnostics and optional agent help stay
available under disclosures. Unsupported packages may require a compatible machine,
not another venv on the same machine.
After stopping an agent or cancelling setup, **Create model environment…** starts
a fresh operation for the retained model on that click. It does not resume the
cancelled helper. Changed source still requires a fresh opening, and late results
from cancelled work cannot select Python or replace current progress.
The selected environment remains available for subsequent openings. Choose it
in tensorViz's environment UI to inspect its path and installed versions.
The tensorViz output channel records creation request/finish state and stage
updates to distinguish a refused request from a running download.
The browser relay accepts the same host-owned creation capability and request,
including restored opening snapshots; it does not accept an interpreter path
supplied by the page or grant the standalone plugin this capability.

This first implementation is native VS Code only. It uses the service Python's
version, supports bounded root `requirements.txt`/`pyproject.toml` declarations,
and installs required model imports plus capture libraries, rather than every
optional training dependency. Unknown import-to-distribution mappings, editable
or local dependencies, unpinned Git sources, private Git authentication, different
Python versions and project-specific CUDA builds still need a compatible project
environment. These limits apply to every chat permission mode.

## Agent setup implementation

The workflow is host state, not assistant prose. `GraphOpeningController` owns the
opening, selected Python check, saved setup and capture result. Verified recovery
is published immediately, before profile discovery or input preparation. Its
result is carried into the next stage rather than leaving the previous missing
Python state in place during a redundant check. A stale or cancelled recovery
cannot publish into a replacement opening.

`packages/contracts/src/graphSetupWorkflow.ts` declares the stage statuses and a
discriminated `GraphSetupNextStep` union. The sidebar, progress list, reminder,
MCP tools and agent all consume that projection. Browser copies are generated by
`npm run protocol:sync`. A source file, edited draft or assistant claim cannot mark
Python ready or attest a completed capture. A deleted TOML file needs saving
again even when the previous captured result remains available.

`CanvasAgent` owns the existing task and selected permission policy. If a provider
reply ends while host setup is still busy, the task retains the original
request/session/source and waits on bounded progress reads. A verified transition
from Python to missing inputs can resume the same provider conversation with
fresh context and a hidden host continuation note. The implementation bounds
continuations; an unchanged input stage may represent a user question and does
not cause automatic retries. Stop, refusal and source/opening changes prevent
continuation. A newly retained failure can resume diagnosis in the same task;
it does not automatically retry the failed operation. Explicit execution review remains a user decision.
Ready means the host has a captured graph, not that capture has merely started.

`extension/src/canvasAgentPrompts.ts` is the single prompt registry, with typed
prompt IDs, versioned definitions and named sections for task/context, Python,
inputs, configuration objects, saving/capture, source review and evidence.
System instructions, conversation formatting, connection preparation and host
continuation each have a dedicated builder. Current source and workflow snapshots
stay attached user data. The Codex and Claude adapters in
`canvasAgentRuntime.ts` use those builders and own provider transport only.
Change prompt wording in the registry; change stage transitions in the workflow;
change approval rules in the host broker. Tool descriptions remain local to their
typed schemas and expose the workflow's next step.

## Dynamic permission explanations

`CanvasAgentPermissionRequest` is the host execution boundary; producers supply
facts from their frozen operation plan instead of permission titles or button
strings. The shared `CanvasAgentPermissionPresentation` is a separate bounded
schema for generated text. The broker clones the request and rejects changes to
facts, duplicate presentation, stale IDs, and approval before presentation.

`extension/src/canvasAgentPermissionTools.ts` exposes `get_pending_permission`
and `present_permission` only on the owning Canvas agent's tool endpoint. The
first returns the current request; the second supplies wording for that receipt
and never accepts a decision, executable arguments, policy or operation changes.
All providers use the shared `permissions` prompt section in
`canvasAgentPrompts.ts`. Exact host facts remain attached data, including when
project metadata contains instruction-like text. Manual/native reviews remain
available without an agent.

The host starts supported environment operations in the background and reports
pending permission data with setup progress, so the same provider can explain
newly resolved package plans. The controller wakes on a new permission even when
the opening snapshot has not changed. If a provider ends its response before
explaining the request, a hidden same-conversation continuation asks for the
missing presentation, at most twice per receipt. Repeated failure cancels that
operation and reports the blocker; it never substitutes a canned permission
explanation or leaves invisible consent pending. Ordinary chat setup uses the
same lifecycle, including operations captured before their first async review.

A new supported operation supplies its namespaced ID, actual facts, effects and
scope to the broker; it does not add a UI template, a new generation prompt or a
policy switch on its name. New execution authority still requires an implemented,
guarded host tool. Machine commands require explicit approval under every setup mode.

## Agent diagnosis and reviewed commands

Model-import verification preserves a bounded structured diagnostic instead of
reducing every failure to its exception type. It includes the actual exception
message, cause chain, traceback file/function/line frames and observed platform,
architecture, Python, Torch and device availability. Source lines, frame locals
and environment dumps are omitted. Known credential patterns are redacted;
truncation is reported. Device observations use Torch already loaded by the
authorized import, and unavailable observations stay null. A macOS host or a CUDA
dependency in a README alone does not prove the cause of a particular failure.

The opening, MCP setup evidence and chat retain that diagnostic. If background
setup fails after the provider's last read, a bounded continuation resumes the
same request and provider conversation to investigate. It does not start another
environment or repeat an unchanged failed installation. Completed command results
likewise remain unread until the agent retrieves them, so an early provider reply
does not turn permission or a command launch into a success claim. Stop, denied
permission and stale task/source context retain their cancellation boundaries.

`extension/src/canvasAgentProjectTools.ts` supplies the same host tools to the
owning Canvas agent across its private MCP sessions:

| Tool | Evidence and boundary |
| --- | --- |
| `get_machine_info` | Platform, architecture, OS release, CPU model/count and memory. No hostname, environment variables or inferred GPU claims. |
| `read_project_file` | An explicit project-relative UTF-8 file, including README, requirements and scripts outside graph dependencies. At most 64 KiB; reports truncation and redaction. Refuses credential paths, binary files and escapes outside the canonical project root. Does not execute the file. |
| `request_project_command` | An exact executable and argument array, project-relative working directory and bounded timeout. Returns a job ID before execution; a host-owned machine-scope request must be presented and explicitly approved. |
| `get_project_command` | Actual job status, exit code/signal, bounded stdout/stderr tails and truncation. Only the same active task can read the job. |

The command tool resolves the executable and canonical working directory before
review and freezes the operation facts. **Every current permission mode asks for
machine commands**, including Full setup access. The agent generates the same
bounded presentation and action labels used for setup permissions; its wording
cannot change the executable, arguments, directory, timeout or authority. Exact
facts disclose filesystem/network access and the absence of a security sandbox.
Provider-native direct shell execution remains disabled. Shell syntax requires
an explicit shell executable and script arguments through this reviewed tool.
Model edits continue through source proposals and their existing review.

Commands use `shell: false`, closed stdin and a filtered environment that omits
provider/service secrets and Python overrides. The default timeout is 30 seconds,
with a 120-second maximum. Each stdout/stderr tail is bounded to 32 KiB; recognizable
secrets are redacted before results reach the agent. One command can wait or run
at a time, with at most eight retained jobs in memory. There is no terminal or
durable command history. Stop, disposal, timeout or a changed task/source context
terminates the process group; context is rechecked during execution.

Typed project-tool and diagnosis sections in `canvasAgentPrompts.ts` ask the
agent to preserve the specific missing module, symbol or native library, compare
project requirements with observed runtime facts and obtain more evidence when
the cause is unverified. The controller waits for pending approvals/commands and
can resume result inspection within the same task, with bounded attempts. It
does not retry a denied command or infer execution from an approval receipt.

Chat shows active generation as four animated dots below the latest message,
with distinct approval, stopped and error icons. The header has no idle circle or
active-work spinner. Reduced-motion users get a static mark; hover/focus and
accessible labels retain the activity detail. Provider connection readiness stays
in agent settings. Its recheck icon immediately disables and spins on request;
**Connection checked** requires observing the matching host check start and a
ready completion. A later acknowledgement from another control does not lose a
check already observed starting. Failure, provider/session changes, configuration
changes and disconnect clear the completion state. Account and disabled-agent
switch explanations remain available on demand. Failed setup keeps an error
state; **Error details** discloses retained
traceback and observed runtime instead of showing a successful ready state.

The shared typed product-scope prompt guides both task workflows toward models,
ML research and related project/software work. It keeps relevant theory and maths,
handles greetings naturally and redirects unrelated work without soliciting its
materials or invoking tools. This is instruction guidance rather than a keyword
classifier or a substitute for host authorization.

## Plugin interpreter roles

The capture worker rejects the legacy module call dispatcher observed in PyTorch 2.0.0 before importing a model. That dispatcher bypasses the instance call observer. This is an unsupported capture runtime, not a reason to replace the project's Torch installation. The chooser and MCP help retain the opening. A successful forward whose canonical capture has `status: failed` also returns a capture error instead of claiming validation; bounded partial captures keep their existing behavior.

MCP opening results check the current selected candidate against the initial environment inspection. After browser recovery changes Python or Torch, results report that candidate's current metadata and omit the obsolete dependency snapshot. This also covers replacement of an environment at the same Python path.

Dependency snapshots also retain the selected environment fingerprint from their inspection. Replacing packages at the same path without changing Python or Torch invalidates that cached report; completed recovery cannot keep reporting an old missing import.

**Browse for Python** resumes the existing opening after a successful chooser result. It cancels obsolete execution without minting a new opening ID, so MCP and help references remain attached to the recovered model. Cancelling the chooser leaves the original setup step available.

Browse and **Select environment** save the chosen Python before capture-library checks. The picker performs bounded inventory discovery only. Missing libraries do not prevent saving the choice or trigger an installer inside the picker. Canvas checks the selected Python in the retained opening, then presents **Add libraries here…** with its fixed progress and result footer. Cancelling selection leaves the old choice intact, and selection alone never claims Python is ready. **Check environment** keeps its existing check and review behavior.

The plugin provisions and reuses one service environment independently from the chosen model interpreter. The service runs graph preparation and supplies pip and packaging; capture, profile resolution and validation use project Python. Metadata checks require only the capture profile, and fingerprints include runtime identity and the installed inventory. The additive installer receives both the manager and the target explicitly. Environment switches keep the service and browser session. Managed fallback capture environments remain isolated per checkout and install only pinned CPU Torch and compatible Pydantic; tool libraries stay in the service. The packaged standalone profile includes `requirements-capture.txt`.

Plugin opening publishes a typed environment step before constructor and input setup. A unique project Python is a suggestion until selected; missing explicit or saved paths remain visible beside bounded project and known alternatives. A ready managed capture environment can appear only for its owning checkout. Selection uses listed IDs and rechecks fingerprints; refresh reads metadata without model execution. Known missing libraries use the same frozen manager and target installer, then re-resolve saved setup in the same opening. Worker and import failures supply MCP investigation context. Unknown imports, package conflicts and unsupported workers do not trigger replacement or automatic retries. Compatibility copies are offered in the extension; the plugin offers compatible environments.

The plugin preserves its existing persistent permission. Its button says **Install missing libraries**, with that permission disclosed; it does not promise a separate VS Code review. Initial permission decline remains blocked until the explicit `request_permission` recovery. No usable Torch candidate returns an explicit interpreter or managed-CPU creation step for the retained model rather than an unsupported native chooser action. Creating a new fallback uses the existing explicit `managedRuntime` opening selector.

## Canvas recovery

**Open as Graph** reveals Canvas before class discovery or service preparation
finishes. Its initial message is **Checking Python environments…**, explaining
the tool service and folder-selected model Python. The clicked file is captured
before focus moves. A view attached after preparation starts receives that same
message, without restart advice or a second service startup. Cancelling model
discovery keeps Canvas and does not open the project instead.

The setup sidebar shares a two-step progress list: Python and Model inputs.
Checks mark host-confirmed completion; draft edits alone never mark a step done.
A compact **Complete setup** reminder appears only while incomplete setup is
closed and idle. It reopens the retained panel, or requests the existing input-edit
flow for a source-only/finished graph, without starting a model run.

Chat permissions are bound to one active turn, source, model and session. They
carry an immutable host operation (`operation`, `scope`, `effects`, `facts`) and
an agent-authored `presentation` (`title`, `summary`, `allowLabel`, `denyLabel`).
Operations are namespaced identifiers, not a closed list of UI templates. The
host validates typed effects and scope for policy; neither generated wording nor
an operation's name grants authority. The expandable Details show the original
structured host facts separately from generated copy.
The host owns a chat-scoped permission mode. **Ask every time** requests each
approval. **Auto setup** permits creation and package installation only for a
host-classified new isolated environment; all source builds and changes to
existing environments still ask. **Full setup access** permits all supported
setup requests. Existing-target additive installs, version replacements and
PyTorch replacements are classified as existing-environment changes, regardless
of interpreter path. A model or renderer cannot classify its own request.

The shield icon beside the composer opens the shared, themed Canvas popover with
three radio-menu choices. It works while a turn is active. The checked row comes
from host state; arrow keys inspect choices without changing policy, Enter/click
requests a change, and Escape returns focus to the shield. A bounded footer shows
the focused/hovered mode's explanation; account and policy details stay on demand.
The menu portals outside the chat's clipping container, including fullscreen.
The icon shows pending acknowledgement without adding a visible status paragraph.
Only a synchronized connected view can change it; the host validates and
acknowledges the selection. Upgrading the mode can settle a current eligible
request after it has an explanation and passes turn/session/source checks. Downgrading governs later reviews;
it does not revoke already granted work. Automatic receipts include the granting
mode and say **Auto-approved**; explicit approvals say **Allowed**. The host
rechecks all normal source/trust/plan guards for either route. Clear/new chat and
cross-model conversation replacement reset to Ask. Modes are not persisted or
read from project files. Manual operations retain their original reviews.

Unpresented requests show **Preparing approval** without consent controls or
attention focus. Presented requests reopen chat with **Needs approval** and neutral heading focus;
minimizing keeps that status visible. A decision is single-use. Stop, Clear,
source/model invalidation, turn end and disposal cancel pending requests. Offline
buttons are disabled and reconnect never replays approval. Approved receipts mean
permission was granted, not that installation succeeded. The history retains up
to twenty receipts. Pending setup reads support an abortable bounded wait
(`get_graph_setup` with `waitMs:10000`) so the agent can wait for human decisions
without rapid polling. Machine commands use the separately reviewed host tool
described above; source proposals retain their existing diff review.

Environment creation and compatibility copies obtain permission before filesystem
preparation, venv creation, resolution or downloads. Resolved package plans and
source builds receive separate permissions under the selected mode; refusing any review stops that operation. A refusal or
aborted agent permission never falls back to a native prompt. Creation failure
rechecks discovery without silently selecting another Python.

Opening checks the saved/configured project Python before importing the model.
When no environment can run the model as it is, Canvas offers agent setup first.
The manual route presents detected environments ranked by what the model still needs. It recommends the one needing
the fewest changes ([ranked environments](#ranked-environments-at-opening)).
Choosing a row sends its host-owned ID; the host rechecks its fingerprint and
remembers it, as the user's own choice, for the model's owning folder. A deleted
saved choice remains visible and never silently becomes a fallback.

Known missing capture/model libraries offer **Review and install missing libraries**.
The host permission broker and exact additive installer retain ownership of approval.
An active agent turn uses an inline chat permission; without an active turn, manual
setup uses the native review. No tool can approve its own request.
Declining or failed repair rechecks the actual environment and retains a failed
opening with the sanitized package diagnostic and a host-owned failure code.
The additive installer reports resolving/downloads, approval, approved source
builds and installation; the opening reports the subsequent capture/model-library
check. These are stages, not measured percentages. A completed install whose
required libraries are still missing reports `LIBRARY_CHECK_FAILED`, rather than
silently returning as successful preparation. Successful repair re-resolves saved
setup and continues the same opening. Package
conflicts remain separate, with a reviewed copy only for supported compatibility
profiles. Unknown import names are not guessed. If Torch is absent in an explicitly
selected project Python, the native exact-wheel review can include it. A model
requiring a specific Torch/CUDA build can instead choose an existing suitable
environment or the explicit managed fallback.

When a binary plan names a missing wheel, additive repair also checks index
availability before asking to build source. `PACKAGE_UNAVAILABLE` retains the
exact requirement and stops before build review. An available distribution still
needs explicit source-build and frozen-install approval; availability is not proof
that the requested version resolves or builds. Temporary plan directories are
removed on every outcome. Manual native review dialogs use VS Code's built-in Cancel
instead of adding a duplicate Cancel item.

An explicit library retry after Stop starts a fresh operation in the retained
opening. Changed source and replayed request IDs still refuse repair; late
cancelled work cannot resume capture.

For a missing model dependency with no supported install recipe, use **Setup
options → Set up project with agent** to investigate the retained opening. Older
manual states can also offer **Use Agent**. The standalone page offers **Copy MCP
investigation request**. The manual screen
labels capture libraries separately from the model check and offers explicit
**Inspect source only**; selecting another row still requires **Use…** before the
host changes Python. Unsupported declarations are recognized as already present
in project manifests. They remain blockers for measured capture, with no guessed
PyPI substitute, automatic Git install or replacement of the managed Torch build.
Source inspection retains its existing limits and never counts as a successful
capture.

When an installed model library conflicts only with the project's own declaration,
for example DINOv2's `torch==2.0.0` with the repository CPU pin Torch 2.14.1 installed ([pins](../local-setup.md#install)), native Canvas also
offers **Try with installed versions** for the saved environment. The model check
reports each conflict's exact declaration (`declaredPins`); the host remembers those
pins for the project and interpreter during the current VS Code session and continues
the same opening. Nothing is installed or written to settings. Choosing the environment
again, or a changed set of pins, starts a fresh check. A failed capture shows its real
error, and an observed import failure brings the choice back. Conflicts with
tensorViz's own capture requirements, such as Python or pydantic, are never
overridable. The extension's MCP endpoint offers the same override as `open_graph`
action `use_installed_versions`, using the current request, opening and source identities.
It is offered only when the environment reports `declaredPins`. The standalone plugin
does not offer this override.

For a saved project environment the same notice also offers **Install in this
environment**, which replaces the installed versions with the project's pins in place
(`replace_versions`). It is never offered for the tensorViz-managed runtime, which every
model shares. The backend accepts only the project's own declarations and resolves them
with pip's dry-run report against hash-verified wheels. Installed packages whose
requirements the new versions would break, such as torchvision for a Torch pin, are
resolved again under any project declaration and join the plan; nothing else may change.
When no wheel exists for this interpreter the plan stops with a plain message, for
example `torch 2.0.0 has no build for Python 3.12 on macOS arm64`. VS Code then lists
every change (`torch 2.14.1 → 2.0.0`, `+ sympy 1.13.1`) for approval and asks separately
before changing Torch, its companions or CUDA wheels. Installation re-verifies the
environment and the project pins against the reviewed digest, then the opening checks
the environment again and continues. There is no backup; other projects using the
environment get the new versions. The standalone plugin and MCP do not offer replacement.

The selected row receives the exact private worker import probe. Successful worker
checks cache by installed environment fingerprint, with at most 32 entries; metadata
and model requirements are rechecked. This does not certify an arbitrary forward.
Runtime missing-import failures return to this step without automatic installation
or retries, while actual constructor/forward failures retain their attempt details.
The expanded Canvas header shows the Python/Torch selection after recovery.
Library planning/review keeps the recovery panel visible over the source preview,
labels preparation separately from a forward capture and retains cancellation.
Unexpected native repair errors are recorded in tensorViz Output; the short setup
message still rechecks actual installed state and does not claim successful repair.
MCP help labels paused environment setup separately from an observed model failure.

MCP returns this same state and a bounded investigation request. `select_environment`
uses a listed `candidateId`; the known additive `install_dependencies` action uses
the opening controller's native review and resumes without replacing the opening.
`refresh_environments` is metadata-only. Agents are directed to inspect worker or
unknown-library failures directly through MCP rather than submit irrelevant inputs.

## Shared opening recovery state

The opening controller can pause before model imports with a typed environment
reason, bounded candidate identities and known missing requirements. Candidate
selection and reviewed repair continue the same opening; source changes and
cancellation discard stale replies. An unsuccessful repair stays paused rather
than starting an automatic retry loop. Refreshing metadata never executes the
model, even when the previously chosen environment is ready.

The wire contract accepts a listed candidate ID, never a browser-supplied Python
path or install command. Constructor submission cannot bypass environment recovery.
Direct MCP investigation requests retain the selected environment fingerprint and
recovery facts alongside the originating model/attempt. This is the shared
foundation; plugin integration remains in progress.

## Ranked environments at opening

An opening checks every detected environment for the model,
not only for tensorViz's capture libraries. A single `inspect-candidate` probe per
environment runs the static model check (imports and project declarations; the
model is not imported) beside the capture check, from one snapshot of that
environment. Rows then say what the model still needs.

Each environment is ranked:
- **Ready:** capture libraries present, and nothing missing or mismatched for the model.
- **Cost:** 1 per library to add, 3 per installed version to change.
- **Not recommended:** unavailable environments, capture conflicts (including Python
  older than 3.10), no Torch, or a model that could not be checked.
- **Ties,** in order: the saved choice, the VS Code selection, the model's project
  folder, the owning folder, previously selected, others. The managed runtime comes last.

How the opening uses the ranking:
- With no saved choice or Python setting, the best ready environment is adopted
  without the picker. If none is ready, the VS Code selection or the only local
  environment is still adopted so its repair can be offered.
- A choice tensorViz saved itself, including any choice saved before this change, gives way
  once to the best ready alternative when it only lacks libraries or has other
  versions.
- A choice the user made never switches automatically: picking a row, Browse, a
  created environment or an approved compatibility copy. Nor does the Python setting.
- The managed runtime is never adopted automatically. A missing or failing Python,
  or an observed import failure, stays visible instead of switching.
- Every adoption still passes the full model check and worker probe.

After choosing **Set up manually**, model-checked environment lists open immediately, including from the header and
when the saved choice is the user's own. When that choice is not explicit, the host
names a `recommendedId`, preselected and marked **Recommended**, even if it is
already saved. An explicit saved choice stays preselected. A search field filters
by name, path, source or version. **Create new environment** follows the list. Candidates without model
information, such as those from the standalone plugin, render as before.

## Detected environments

The native picker now shows bounded metadata rows rather than paths alone.
Discovery uses the actual model resource and project root, then the owning/other
workspace folders, the Python extension's known environments, active venv/Conda
paths, previous explicit selections and the ready managed fallback. It never
downloads a runtime merely to populate suggestions or scans home directories.
Remote windows collect paths and run probes in their workspace host.

Rows distinguish **Used by tensorViz** from **Selected in VS Code** and show Python,
Torch/build and capture-library presence. A missing explicit saved/configured path
stays visible; another working row does not replace it. Presence means distribution
metadata was found, not that the model imports or forward succeeds. The chosen
environment still receives the actual worker import probe before acceptance.

The shared collector inspects at most 16 candidates, with at most two concurrent
stdlib metadata probes and a 15-second per-process metadata deadline. Failures stay
on their own rows. Aliases resolve by environment prefix; separate venvs remain
distinct even when they share a base Python binary. Saved choices precede suggestions,
and a ready managed runtime stays available as a labelled fallback. Cancellation
refuses obsolete probe results and does not display a picker afterwards.

Capture metadata includes a fingerprint of interpreter/runtime identity and the
installed inventory. It changes after package or virtual-environment configuration
changes; deeper worker-probe caching can therefore be invalidated by actual target
state. The current native picker refreshes metadata on every opening. Selection
notifications no longer block continuation while waiting to be dismissed.

## Extension service and project Python

Python selection is scoped to the model's project root, including nested projects;
files without a nested project use their owning workspace folder. The picker
queries the Python extension with the actual model resource, so its suggestion
can differ between files and roots. Browser/MCP recovery and compatibility-copy
promotion pass that same resource. A saved unavailable path remains authoritative.

The old workspace-wide choice migrates only to the first folder, which owned the
old picker. It still applies to models rooted in that workspace folder, while nested
projects require their own choice. The former machine-wide managed
default no longer overrides a new folder's Python selection; the ready managed
runtime remains available as a fallback without another download. Scoped selections
are host-local and are not Settings Sync data.

Extension startup now uses the backend-role resolver and its managed runtime,
independently of the saved capture choice. Its development fallbacks do not consume
`tensorviz.pythonPath` or the Python extension's active model environment. Declining
service provisioning does not silently execute a fallback interpreter.

Selecting project Python checks `requirements-capture.txt` and imports the actual
private capture worker. It does not require FastAPI, Uvicorn, LibCST, dotenv or
websockets unless the model itself imports them. A pipless selected environment
uses the service interpreter's installer. Changing project Python cancels opening,
validation and source-edit work, and resumes the retained model without disposing
the backend or reconnecting its views. Browser and MCP model dependency inspection
also use the service manager with the project target explicitly bound.

## Interpreter roles in the shared installer

`DependencyInstallOptions.python` is always the project capture interpreter.
`managerPython`, when supplied, is the tool-owned interpreter that provides pip
and packaging. Planning, source building and installation use this manager with
an explicit `--target-python` helper argument. The target does not need pip or
packaging for inspection or reviewed installation.

The helper reads the target's distribution inventory, top-level import presence,
Python version, platform and architecture using a bounded stdlib subprocess.
Installed packages in the manager cannot satisfy the target's model imports or
constrain its Torch version. Static model import discovery includes real parent
package initializers, optional-import rules and the tested
`torchmetrics → torchmetrics`, `cv2 → opencv-python` and SciPy mappings.

New plans freeze the target executable/prefix, Python ABI/architecture, executable
file identity, virtual-environment configuration, installed distributions and
requested requirement bytes. Existing versions remain fixed. Apply rejects a
changed plan, target or request; approved wheels install with exact hashes and
without dependency re-resolution. The target prefix is used explicitly.
An external manager refuses installation into unwritable, externally managed or
ordinary system Python targets; a writable project venv or Conda environment is
required. A cancel can leave some approved additions installed, so recovery must
reinspect rather than claim a rollback.

Compatibility cloning and version repair retain the existing explicit marked-copy
boundary. The target's environment is copied, never the manager's. Model import
verification executes in that copy. Source package building retains its separate
approval and subsequent wheel review.

## Capture requirement profile

`backend/requirements-capture.txt` declares Torch and Pydantic separately from
the existing server requirements. `inspect-capture` checks this profile. The private
worker import probe uses the same bootstrap as capture. Profile inspection alone does not prove an arbitrary user model
captures successfully.

## Direct MCP help for failures

The full opening panel and compact failure notice expose **Get help through MCP**.
This opens the existing connection guide with a copyable investigation request,
without automatically messaging an agent or retrying the model. The snapshot
identifies the model file/class, opening ID, source digest/version, failure kind,
bounded detail and whether capture evidence exists. The extension also reports the
selected interpreter and source when help opens; the request instructs the agent
to verify the current environment and source before acting.

An already enabled MCP endpoint is reused. Failure help opens the guide directly;
agent setup remains an explicit control. The guide reports actual connection
receipts separately from configuration. Reopening updates its snapshot; later
mutations cannot change the request already presented. Closing and returning to
Canvas does not execute the model. Remote SSH retains explicit forwarding.

Standalone Canvas copies the same investigation request into the clipboard. MCP
`open_graph` includes `helpRequest` on failed openings. These describe observed
failure context, not proof that the tool or model is fixed. No chatbot is added.

## Environment selection redesign

### Automatic readiness on the extension home

The native home checks only the owned tensorViz runtime automatically in trusted
windows. Project Python discovery and capture metadata are deferred to model
opening; a VS Code Python selection cannot mark home unready. Home does not
install packages, save choices or import the model. At an
explicit opening, an environment that can already run the model is adopted without
the picker. Otherwise a compatible VS Code selection or sole project/owning-folder
environment is adopted. The user's own saved choice and the Python setting remain
authoritative even when broken; a choice tensorViz saved can give way to a ready
environment ([ranked environments](#ranked-environments-at-opening)). Ambiguous
candidates with nothing ready still require selection. The chosen Python is
freshly checked for model libraries and worker readiness.
This extension entry behavior does not change standalone plugin selection.
See [one starting screen](canvas.md#automatic-readiness-and-one-starting-screen).

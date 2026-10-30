# Source-reviewed graph editing

## Contents

- [Grouping transform](#grouping-transform)
- [Insertion name safety](#insertion-name-safety)
- [Using source proposals from the local page](#using-source-proposals-from-the-local-page)
- [Authority and transport](#authority-and-transport)
- [Validate after editing](#validate-after-editing)
- [Shared source writer](#shared-source-writer)
- [Reviewed project declarations (F4 host gate)](#reviewed-project-declarations-f4-host-gate)
- [Host proposal API](#host-proposal-api)
- [Captured numeric literals](#captured-numeric-literals)
  - [Captured constructor arguments](#captured-constructor-arguments)
- [Captured configuration definitions](#captured-configuration-definitions)
- [Captured tensor reconnection](#captured-tensor-reconnection)
- [Insert or remove a captured functional activation](#insert-or-remove-a-captured-functional-activation)
- [Run the reviewed candidate before Apply](#run-the-reviewed-candidate-before-apply)

Python source remains authoritative. Graph selections produce proposals for
native VS Code review, explicit Apply and reprojection of the changed buffer.
The browser does not write source or keep an independently edited graph. MCP proposals use the same review boundary.

## Grouping transform

`/api/ide/apply` accepts one `group_layers` operation with `node_ids` and a new
`node_id` for the block. It extracts a consecutive unary chain into a small
named `nn.Module` class in the same file. A name collision gets a fresh class
suffix; existing classes are preserved. Child names survive under the new
block, for example `fc1` becomes `features.fc1`.

Grouping changes module/parameter paths and therefore state-dict keys. Existing
checkpoints may need a key migration. The source review must disclose this.

The supported chain has consecutive declarations, literal constructor arguments,
and one forward use per layer, in declaration order. Forward may use nested
calls or consecutive assignments passing the same tensor. Branches, reused
layers, dynamic attribute access, references elsewhere, expression-based
constructors and mixed grouping/edit batches are refused. The service returns
patch ranges and never imports or writes the user's model.

The transform retains comments and unrelated source. Its generated class uses
the existing multi-file projector's module hierarchy, so accepted source can
produce a collapsible block without an extra graph authority.

## Insertion name safety

Insertion refuses existing attribute names and invalid Python identifiers before
creating a patch, preserving existing declarations.

## Using source proposals from the local page

Select the relevant nodes and describe the change in Canvas chat. Mapped file:line links still open the source directly. Nodes no longer expand into inline editing forms. Deterministic MCP `propose_edit` recipes remain available, and the Canvas-owned agent can prepare a complete source proposal across up to 16 existing, graph-recorded workspace Python files.

The agent reads current source and exact digests before proposing replacements. A draft can be revised without losing earlier pending files; removing one pending change must be explicit. **Discard** removes the entire draft. Native VS Code diffs show every proposed file. Accept/Apply makes one undoable editor edit, saves only the accepted files, and opens the model with its current saved setup. A failed save or intervening edit stops execution. Preparing or revising a draft neither writes source nor runs the model.

Captured source facts, proposed shape effects, and actual candidate results stay separate. A failed refresh marks retained graph evidence stale. Shared-class changes affect every instance of that definition; changes to parameter names or grouping can also change checkpoint keys. Inspect the full Python diff before Apply.

## Authority and transport

`sourceEditRequest`, `sourceEditCancel` and `sourceEditResult` reuse the authenticated
view/broker/host channel and its request ownership. Browser requests carry bounded
typed intent, semantic node identities and the current projection identity. They
carry neither source text nor an arbitrary source path. The host resolves its own
span map and calls the existing code-apply service using its private launch token.
The broker routes results only to the requesting view and synthesizes cancellation
on disconnect. Cross-view cancellation and mismatched results are refused.

The capability is `canProposeSourceEdits`; direct source/graph write capabilities
remain false and direct canvas mutation guards remain active. One native review
may be open at a time. The host checks Workspace Trust, the target, every projected
dependency's buffer digest and document version before planning and immediately
before applying. A mismatch refuses the edit and preserves the user's changes.
Returned patch ranges must reconstruct the exact reviewed digest.

Captured edits can also depend on saved binary fixtures, checkpoints and explicitly
tracked resources. Review hashes their raw bytes in bounded chunks without opening
them as text, importing code or deserializing data. File/directory identity, file
size, change timestamps and unsaved-buffer state remain guarded through the final
writer check. Changed or replaced resources, symlinks, conflicting source roles,
missing files and cancelled reads refuse. Limits are 128 resources, 2 GiB per file
and 2,144 MiB total, independently of the source text limits. The resources remain
unchanged; this freshness check does not run the proposed model.

If VS Code closes a source/dependency document while showing the native diff,
Apply reopens it and verifies the reviewed bytes before using its new live document
version. Closed document objects can retain old text and are not freshness evidence.
A changed file, failed reopen, or another close during the final read refuses the
proposal. This guard is shared by static, captured and project-declaration edits.

A same-class reprojection can arrive before the apply receipt. The browser keeps
that request pending until the host answers, rather than reporting a false
cancellation after a successful edit. A dropped connection reports uncertainty and
asks the user to inspect the VS Code buffer before retrying.

## Validate after editing

Save the model and any changed dependencies in VS Code. Validation runs automatically by default; **Refresh graph** also requests a run. The selected Python environment runs the model in a
trusted workspace; progress and the result appear beside the graph. **Cancel**
stops the check. A source change invalidates the result, so validate again after
editing. This checks one traced input and branch, not all inputs or training.

Set `tensorviz.traceOnSave` to `false` to require a manual run after saves. Typing only refreshes the source projection; **Inspect source only** also runs no model code. A missing environment, constructor arguments,
execution failure or stale source produces an explanation instead of a pass.

A matching saved model profile uses its `limits.timeout_ms`, including values
above 225 seconds. The accepted range is 100–86,400,000 ms and the profile default
is 60,000 ms. With no matching profile, automatic forward validation remains
bounded at 180 seconds. These limits cover the worker execution; profile
selection, environment checks and graph refresh are separate. A timeout stops
the worker and reports a failed check. Cancel, source changes and session closure
still stop a long run, and a late result cannot turn that cancellation into a pass.

A failed trace merge also marks retained evidence stale and clears measured editor
inlay hints. Previous Passed/Failed receipts stop certifying the stale graph;
the current attempt's refusal or cancellation explanation remains visible. The
graph retains its prior shapes and resolution diagnostics as historical evidence.

## Shared source writer

Native review uses provider-independent source artifacts and `SourceWriter` in
`extension/src/sourceChanges.ts`. Deterministic proposals need no model call or
provider credential. The writer checks live-buffer digests and the caller’s
revision/cancellation guard before applying one undoable workspace edit.

## Reviewed project declarations (F4 host gate)

Project `node`, `edge` and `experiment` declarations use the same native full
Python diff, toolbar Apply/Discard, `SourceWriter` and review reservation as layer
recipes and MCP proposals. A declaration cannot start while another proposal is
preparing or under review, and it cannot replace another review's toolbar commands.
Apply submits one undoable `WorkspaceEdit`; native Undo restores that edit. Discard
writes nothing. After Apply, the host automatically re-indexes the current buffers
and publishes the updated project index and graph, retaining the focused experiment
and expansion set when still present. No save or reopen is required.

The browser supplies intent and an identity to compare, never a filesystem
destination. The host requires the exact active workspace, project root,
experiment, index digest and revision, then resolves the entrypoint and allowed
edge labels from its own index. The index retains the digests of the source bodies
it admitted. A missing entrypoint, unknown edge label, invalid declaration or
missing dependency evidence is refused. Before proposing, before opening review,
and immediately before Apply, the host checks Workspace Trust, the active project,
and the entrypoint/import closure's live source digests and captured document
versions. Newer edits are preserved, including changes during the final document
read by the shared writer.

Declaration fetch and response decoding have a 10-second deadline. Retargeting a
class/project, changing experiment focus, re-indexing, host disconnect or disposal
cancels pending declaration work; a late response cannot open a diff or apply.
Source and candidate sizes are bounded, and the candidate must name the captured
entrypoint and before digest. VS Code notifications surface refusals, cancellation
and success without backend error bodies, source text or credentials. An uncertain
editor write tells the user to inspect the buffer before retrying.

The wire contract enables declaration traffic through both transports and
the authenticated broker. The existing shared review guard remains the sole write
path. Host regressions drive the real transport, review toolbar and writer with
VS Code primitives stubbed, including refresh after Apply.
The same review guard is the only path that writes Python.

## Host proposal API

The browser and external tools share one source-proposal operation. It reports
preparation, the native review title/file/digests and its final outcome without
exposing a write endpoint. Only one review can be active per window. Cancellation,
disconnection or a 30-minute expiry closes the pending review and prevents later
application; all source/dependency checks still run immediately before Apply.

Source-import errors direct users to valid `nn.Module` Python source; there is no built-in chat fallback.

## Captured numeric literals

Ask Canvas chat to change the selected captured operation, or use a listed `propose_edit` capability.
A proven `tensor + number`, subtraction, multiplication or division can
offer **New literal** with an explicit integer/float type. The proposal explains that
the edit changes every execution of the authored expression, including unobserved
uses; it cannot specialize one loop iteration or one shared class instance.

**Review in VS Code** opens the complete native Python diff. Apply updates and saves the accepted source, Discard writes nothing, and native Undo restores the edit. A no-op
preserves exact source spelling. Preparing the diff does not execute Python.
When its displayed capture has a retained resolved setup, the native diff offers
**Run reviewed candidate** for an explicit check of the copied proposal. Save and
use Validate to check the live source. Computed expressions, ambiguous
source locations and unsupported derived operations remain inspection-only. Legacy
parameter/add/delete/group intents cannot bypass this capability boundary.

A proven functional activation supports replacement with
ReLU, GELU, SiLU, sigmoid and tanh. It changes only the callee, preserving the input
expression. Current support requires a proven functional namespace or direct import,
one positional or keyword input and compatible literal options; shadowed/rebound names and unsupported
calls explain their limitation instead of dropping arguments. Shared scope and
native review are the same as for captured literals.

For a captured module activation, the recipe replaces its verified
constructor (for example, `nn.ReLU()` becomes `nn.GELU()`). Attribute names and
forward calls stay unchanged. The proposal discloses shared instances and calls;
compatible options remain intact. Explicit `inplace` restricts replacement to
ReLU/SiLU, preserving proven mutation when true; explicit GELU approximation
permits only a no-op. Modified forward/hooks, ambiguous sites and incomplete
recording refuse the recipe.

Directly imported functions/classes keep their shared import and other uses.
When no usable namespace exists, the review adds a uniquely named Torch namespace
import together with the selected callee change. Both changes apply in one
WorkspaceEdit; a no-op adds nothing. The proposal discloses the required import.

Additional supported recipes are described below. Native Undo remains the
source-edit mechanism; structural Undo and Redo are not a separate review action.


### Captured constructor arguments

A captured module call can expose directly authored numeric arguments in a verified
constructor, including a custom class when capture verified its actual class identity.
Request the constructor argument change through Canvas chat or its listed MCP capability. Positional fields use **Argument 1**, **Argument 2**,
etc.; keywords retain their source names. The proposal shows the current source value,
shared scope and shape/checkpoint implications. Variables, configuration references,
computed expressions and general factory calls need separate supported recipes.

Apply changes only the selected numeric token in one undoable WorkspaceEdit. The
native diff preserves other arguments, comments and call sites. A failed forward
can offer a constructor repair, and an eval Dropout call can expose `p` even when
it records zero operations. These are syntax-checked proposals: they do not run
the model or fix downstream layers. Save and Validate to check the resulting model.

## Captured configuration definitions

A captured module can offer **Edit** for a constructor argument backed by
a unique same-file local/module numeric definition. The capability distinguishes
`Argument 1 = width (4; edit definition)` from a direct literal. The proposal shows the definition location, known references, observed constructor sites and sharing.
Choose a replacement and review the native diff: it changes `width = 4`, preserving
`nn.Linear(width, ...)` and every other reference. It affects all references to
that definition, including uses not captured in this run, and can change shapes
or checkpoint compatibility. Other source is not repaired automatically.

An Apply receipt remains inspectable after source refresh.
Parameters, imports, attributes, ambiguous writes and computed definitions are not
supported by this recipe. Save and Validate to inspect the resulting execution;
preparing source review does not execute it.

With a retained execution setup, **Run reviewed candidate** checks the requested
value at every observed direct constructor argument for that definition, including
retained instances that were never called. Its result lists the number of checked
arguments and initializer entries. Computed or expanded constructor references
remain unverified, as do other uses and later behavior. A successful value check
does not certify later edits to the applied source.

## Captured tensor reconnection

Request a reconnection for a captured binary operation, identifying operand 1 or 2 and an available tensor from the listed capability. Each choice shows the variable's definition line. In
`b = a + a`, changing only operand 2 to `h` proposes exactly `b = a + h`.
The selected use is explicit even when both operands read the same tensor.

Choices come from verified source order in supported straight-line code. Later
results and overwritten values are unavailable; mutation, control flow and
ambiguous source refuse the recipe. Availability does not establish compatible
dimensions. Review shows the exact name change and its effect on every execution
of the source expression. Apply saves the accepted source and opens the model; inspect the run to
inspect the new computation. No other layer is repaired automatically.

A source refresh does not hide an outstanding Apply receipt. The limits are the
straight-line rules in this section: later results, overwritten values, mutation,
control flow, and ambiguous source are refused, and availability does not prove
compatible dimensions.

## Insert or remove a captured functional activation

Request an activation on operand 1 or 2 through Canvas chat or the matching MCP capability. This wraps only that named operand: selecting operand 2 in `a + a`
can propose `a + F.gelu(a)`. The proposal discloses any new Torch import included in
the native review. Other uses of the tensor stay unchanged.

For a supported functional activation, request its removal. The proposal shows its original input, and review replaces only the call with that input.
For example, `h = F.gelu(input=h)` becomes `h = h`; assignments and later tensor
uses remain in place. Removal cannot choose a different tensor.

These recipes require a proven straight-line scope. Removal supports
single-line functional calls with named inputs; in-place activations and computed
or multiline inputs refuse. Both changes affect all executions of the authored
source, including shared instances. Review checks syntax; save and Validate to
observe the changed model.


## Run the reviewed candidate before Apply

For a captured proposal with a retained resolved execution plan, use **Run reviewed
candidate** in the native diff toolbar. Opening the review itself runs nothing.
The host executes the exact reviewed file in a temporary workspace using the same
selected interpreter, recipe/profile, input and state setup. The worker is separate
from live Validate, so its output cannot replace live graph measurements.

The full diff stays pinned while **Inspect candidate result** opens a read-only
result beside it. It reports static checks, runtime outcome, freshness, capture
coverage, actual input and pre-call model-state equivalence, and cleanup. Missing
identity is unavailable, not equivalent. A successful forward with partial capture
remains partial. Supported recipes report observed semantic postconditions separately;
missing or incomplete evidence remains unverified. Only fixed diagnostic codes appear
here, not arbitrary printed exceptions.

**Cancel candidate run** stops this execution and leaves review open. Apply or
Discard also cancels a pending run and waits for cleanup. Apply still needs exact source/dependency freshness and makes one undoable editor edit before saving the accepted files. A failed forward
does not prevent ordinary source editing; the outcome states that the copied
candidate failed. A cleanup failure refuses Apply and warns that temporary files
may remain. Native Undo remains the source-edit mechanism.

Automatic input-search captures without a resolved replay recipe do not expose
this Run action. Browser and MCP offer the same candidate controls and retained
results. After Apply, **Check applied source** compares
the confirmed receipt with the current source and a later live capture; Save and
Validate remain explicit actions. Native Undo remains the source-edit mechanism.

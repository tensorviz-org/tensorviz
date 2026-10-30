# Review model changes in Canvas — reference

<a id="review-model-changes-in-canvas"></a>

For the short walkthrough, see [the user guide](../graph-diff-review/README.md). This page keeps the detailed behavior and implementation notes.

## Contents

- [Coverage and correspondence](#coverage-and-correspondence)
- [Input connection changes](#input-connection-changes)
- [Source operation cards](#source-operation-cards)

Use **Graph diff** on an open native Python code diff, or **Open PR** in Canvas
or the Command Palette. Paste the PR URL or choose a repository PR; TensorViz
reads its revisions, discovers affected models, and opens the comparison without
requiring a live model. See [direct PR review](../graph-diff-review/pr-workflow.md) for supported source
providers, model selection, authentication, update handling, and limits.

Source Control's **Graph diff** uses the same Canvas: Changes compares the index
with working files (including unsaved buffers); Staged Changes compares HEAD with
the index. **Graph diff against HEAD** in the editor or Command Palette compares
HEAD with working files. Its existing `tensorviz.compareWithHead` command ID is
retained for keybindings. The retired ghost-comparison route and parser are removed;
these actions neither open the old editor canvas nor create `.tensorviz/views` files.

For advanced local comparisons, open a model with **Open as Graph**, then select
the Git comparison icon or run **tensorViz: Review model changes**.

- **Working changes:** HEAD versus current Python files, including unsaved buffers.
- **Branch changes:** the merge base of a chosen base branch and HEAD versus HEAD,
  for a checked-out pull request. Fetch refs first; this command only reads local Git.
- **Two revisions:** two local branches, tags, or commits.

Opening a local review from a captured model keeps the full graph on Canvas.
Its nodes, connections, expansion and arrangement stay intact. The change
navigator reports source edits, including constructor-default edits on the root.
An edit is placed on a captured call only when its registered-path correspondence
belongs to an exact reviewed source snapshot; unmatched changes focus the root
and say **Not located in this capture**. The capture is reference context: its
shapes and connections have not been re-executed, and no runtime Before/After is
inferred. The toolbar shows the current change in a compact row, for example
`ffn_layer default: 'mlp' → 'swiglu'`. It has no graph-type switch or capture
explanation. A PR or another model without a matching local capture uses the
source comparison below.

Source modifications on nodes and module frames have an amber outline and tinted header, including
collapsed modules and the large-graph overview. The outline stays two screen pixels
wide when zooming out; selecting the node retains its diff highlight. Connection-only
changes use blue and say **inputs changed**; source plus connection changes say
**modified + inputs**. The counts and navigator distinguish these cases, including
model outputs. See [input connection changes](#input-connection-changes). Constructor
default edits highlight the model root, not unproven internal calls. Errors keep
their red outline. Closing review removes the highlights.
When a reviewed declaration is absent from source lowering, its exact registered
capture path can still highlight the individual module. Added declarations are
green on an after-side capture, removed declarations red on a before-side capture,
and modifications amber. All observed calls of a uniquely registered module are
highlighted. The navigator lists each declaration separately; selecting a mapped
call shows that change and opens its source span. No runtime topology is inferred.
Without a matching capture, an omitted declaration remains in the navigator and
focuses the model root. Unobserved or ambiguous calls are not guessed.

The native code diff stays in the editor and the graph stays in Canvas. Before,
Diff, and After share tile positions. Diff shows green additions, dashed red
removals, amber modifications, blue input changes, old → new parameter values, and replacements in
one tile. Removed edges curve around inserted tiles. Previous/next visits each
changed tile; source links open the correct revision and mapped line. Moving the
cursor within a mapped code span focuses the corresponding tile. Collapsed
modules retain a summary. Closing review restores the live graph and arrangement.

Review chrome takes two rows so the graph keeps the height. The page header shows
**Graph diff** with the compared revisions (prefixed by the review title, when one
exists) and holds the code diff, **Refresh graph diff**, and close actions. The
toolbar row holds Before/Diff/After (source comparisons only), the previous/next
stepper, the current change with a status-coloured dot, stale or PR-updated notices,
the counts, and the coverage disclosures. Long change text truncates with the full
text in its tooltip; narrow panels wrap the counts instead of adding fixed rows. A
multi-model PR adds one row for its affected-model chooser.

Both code sides are immutable snapshots. Editing a working-tree snapshot's Python
source marks review stale; **Refresh graph diff** reads the revisions again.
Review never checks out branches, modifies the index, writes model source, or
executes either revision. A capture made while opening the live model is not
reused as evidence about these snapshots.
Review projections are stateless (`review_only`) with a separate graph identity:
they do not write to the project store or advance the live graph's revision.

## Coverage and correspondence

This surface is available in **native VS Code Canvas**, and compares **source
projections of the selected model** using the existing bounded multi-file AST
projector. Reviews on an existing capture have a compact **Graph diff** footer;
standalone source reviews say **Source comparison · Not executed**. Constructor
sizes are parameters; runtime shapes are not fabricated. Source notes and
truncation remain visible. Canonical execution-capture comparison is separate.

Coverage notes prioritize changed files and the selected model's known dependencies.
Identical before/after notes appear once. **Coverage limits** and **Other repository
notes** are neutral, collapsed disclosures; only one opens at a time, with bounded
scrolling. Tab into the details and press Escape to close them and restore focus.
Other notes concern discovery of other affected models. Optional external imports
alone do not produce warnings; unresolved dynamic references and imports name the
actual source line and reason. Review shares CodeLens's declaration-scoped module
discovery, including Torch container subclasses and proven fallback aliases.
Repository/source budget limits remain in the main coverage notes.

Named declarations match by qualified attribute identity and defining file.
Anonymous statements can match an edited expression when the parser proves a
unique local assignment in that method and exactly one projected operation for
the statement. Reassigned names and statements with multiple operations do not
receive this binding identity. Other statements require a unique exact source
digest in the same file and parent scope. Generated ordinals, positional container entries, tensor shapes,
and topology are not correspondence evidence. Unmatched named layers are reported
added/removed only with complete projections; otherwise they say **unresolved**.
Edges involving unresolved identities are not counted as confirmed additions or
removals. Endpoint/port multisets preserve duplicate value uses. Line movement,
canvas positions, and generated edge IDs are ignored.

Opaque Python cards name dotted calls in AST evaluation order, so
`F.normalize(x.float())` reads **float → normalize**. The changed tile and source
link retain the exact expression. This is a source description, not a measured
dtype or execution trace. Expressions with conditional evaluation and older graph
files without order metadata use an unordered list instead of misleading arrows.

Source-proven Torch import aliases and equivalent positional/keyword constructor
arguments compare by their meaning. Unsupported constructor expressions remain in
the comparison, so a changed expression still produces a modification. Literal
scalar `+`, `-`, `*`, and `/` operations retain their operand and its side. A numeric
constructor attribute used by `forward()` is included only when source shows one
unconditional assignment and no later write; mutable or dynamically assigned
attributes are not inferred. Execution, shape inference, and export preserve these
scalar operands. Dedicated parser, projection-identity, runtime, and export
regressions cover these rules, including private capture-worker imports.
Omitted literal Torch constructor defaults compare equally to explicit defaults.
Default edits retain both values (for example, `bias: true → false`). RNN, GRU,
and LSTM use their Torch base signature; unpacked arguments receive no assumed
defaults, and unproven user constructors cannot borrow Torch defaults.

Repeated value uses survive parsing, node renaming, inheritance composition, and
multi-file expansion. For example, `layer(x) + layer(x)` retains two incoming
connections to the addition. Repeated generation of the same module bridge is
still deduplicated; edge identities distinguish actual operand occurrences.

Direct, source-proven local instance helpers are expanded from `forward`, including
nested helpers and transparent `*args/**kwargs` forwarding wrappers. Argument
binding preserves literals, keywords and literal tuple/dictionary unpacking;
tuple/dictionary returns route their selected producers. Inline Torch module calls
such as `nn.ReLU()(x)` become individual operations with their actual source spans.
Each helper call keeps its own local bindings and comparison context. Recursion,
decorators, rebinding, generators, comprehensions, unknown argument expansion and
the depth/work budget stay opaque. Source paths are not an execution trace.

Within the same proven call context, pure operation insertion/deletion is reported
only when the unchanged operation sequence has a unique exact source embedding.
This permits two identical new ReLUs around unchanged operations while refusing
ambiguous repeated anchors, replacements and reordered sequences. Truncated
snapshots cannot establish this proof.

An opaque `self.helper(...)` call retains the source bodies of reachable methods
on the same class. Editing a reachable body marks the call as **Helper
implementation changed**; unrelated methods are excluded. This records source
changes without executing helpers or claiming that an opaque body is understood.

The enclosing `if`/`else`, `while`, and `for` conditions are retained on guarded
layers and operations. A changed predicate marks the affected tile as **Execution
condition changed**. This does not evaluate the predicate or resolve dynamic
control flow; the comparison remains source-only.

Matrix products written with `@`, `torch.matmul`, or `torch.bmm` remain explicit
operations. Attention stages connect through these operations instead of copying
all previous ancestors into later edges. The four-stage regression checks two
inputs per matrix product, bounded edge growth, and faithful code export.

Constructor defaults are compared separately from projected layers. A default
such as DINOv2's `ffn_layer` changing from `mlp` to `swiglu` appears on the model
root even when dynamic blocks cannot be expanded statically. A changed class
body with no mapped layer change remains unresolved and links to the code diff;
it does not produce a no-changes claim. Comments and docstrings are excluded from
this class comparison. Incomplete projections say **Comparison incomplete**.

Review also retains direct `self` declarations from the initializer, including
conditional expressions and factory calls omitted by graph lowering. Each has
its own source digest and declaration span; branch/loop conditions participate
in that digest, and repeated assignments remain ambiguous. Review responses retain
all supplied source-file digests so a capture can be checked against the complete
reviewed snapshot. Captured registered module paths remain available for matching
even when no static layer card exists. Shared aliases do not prove a unique mapping.

This does not prove behavioral equivalence or represent every Python edit as a
structural change. Dynamic construction, control flow, external code, and opaque
containers retain the projector's limitations. Both revisions must contain the
same selected file and class; whole-model additions/deletions and file/class
renames require choosing a common target or inspecting the native code diff.
For GitHub pull requests and selection among affected models, use [Open PR](../graph-diff-review/pr-workflow.md).


## Input connection changes

Inserting a layer in front of an unchanged operation changes what feeds that operation. It does not edit the operation's own expression. Review keeps those facts separate:

- **inputs changed** (blue): incoming connections changed; the matched node's projected source and parameters did not.
- **modified** (amber): the operation, expression, constructor arguments, execution condition, or local helper implementation changed.
- **modified + inputs** (amber): both changed.

A node with both kinds of change counts once as modified. Before and After clear these badges. Edges that touch an unresolved node are not confirmed connection changes. A captured overlay needs an exact source snapshot. It labels the source change and does not invent a new execution.

## Source operation cards

Graph review and the live graph use the same layer cards. Source projection keeps `reshape`, `view`, `permute`, and source-proven `interpolate` calls, including nested chains. Only the tensor operand becomes a tensor connection. Shape and dtype reads stay scalar metadata. Unevaluated options such as `self.interpolate_antialias` are shown as source expressions and compared as parameters. They are not measured shapes. Export refuses symbolic source arguments rather than dropping them or inventing defaults.

An expression card is titled from the call that produces the statement's result. `torch.where(...)` reads **Select by condition**, with `torch.where` under the title. Nested calls that only prepare arguments are not drawn as a single chain. Unknown calls keep their method or function name.

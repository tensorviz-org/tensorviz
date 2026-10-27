# Setup and inputs

## Contents

- [Model setup layout](#model-setup-layout)
- [Unchanged calls](#unchanged-calls)

The Canvas form and the saved `tensorviz.toml` profile. Opening steps are in [open a graph](graph-opening.md). Agent-filled inputs are in [connect MCP](mcp.md#fill-missing-inputs).

## Model setup layout

The setup panel uses an opaque Paper/Ink surface so graph controls underneath do
not show through. Its header keeps the selected
class, file and opening status visible. **Model** groups constructor/configuration
values and the custom-factory option; **Example inputs** groups tensors, literals
and Python input functions. Defaults and configured values keep their existing
disclosures, while missing required values stay exposed. **Run settings** contains
mode, device, gradients and Python-environment selection; **Advanced setup** keeps
configuration overrides and explicit JSON bindings.

Fields scroll within the panel; **Open graph** stays in a separate footer. It is
still bound to the same validated form, so an empty or invalid required field
blocks capture and opens its containing disclosure. **Edit tensorviz.toml** and
**Use Agent** stay together; neither submits the draft. Escape/Close returns
to the retained Canvas. Busy states preserve cancellation and reduced motion.
In short windows, the panel omits its repeated file line (still in the Canvas
header), bounds the status text and trims the footer hint to preserve field space.

The setup panel keeps **Edit tensorviz.toml** and **Use Agent** beside the form.
Use Agent starts an automatic setup-and-capture task in the built-in Canvas chat,
using the same installed agent and settings as other Canvas requests. It attaches
the exact model, source, missing fields and environment; the agent reads the setup
contract, submits a complete recipe and follows capture to completion. No copied
prompt, MCP connection guide or separate request in another chat is required.
Progress and Stop appear in Canvas. Failed or partial captures remain incomplete,
and missing information or required native approval is reported explicitly.
The form and hand-authored profiles still work independently. Unsaved form edits
remain in the form; the agent reads saved setup and source. The standalone plugin
continues to copy requests for its surrounding chat.

**Edit tensorviz.toml** opens the selected profile. If the file is absent, the host
offers a reviewed, unsaved model-specific starter with constructor and input
bindings left unresolved. Existing content is preserved. Saving a complete profile
while this setup is waiting resumes opening; opening the file alone never runs it.
After a successful capture, the header shows **Saved as** with the profile name
and **Open tensorviz.toml**, alongside the existing **Change inputs** and
**Refresh graph** controls. Disconnecting the agent does not remove the profile.

Setup uses the graph's existing theme and keeps unresolved fields together. Filled
values stay editable in collapsed settings; typing never collapses the active field.
Failed captures expand their required fields again. **Example setup · sources and
assumptions** exposes the recipe's file/line evidence and synthetic choices beside
the form and captured-input summary. Source defaults, optional inputs, tensor
generators and execution options expand in place. Each tensor shows its dtype and generator before execution. **Open graph**
saves and runs the model; recovery actions live under **More options**. Unknown or purely
variadic forwards without a supported constructed-model input contract require
explicit inputs or a deliberate empty-call selection.
Declared zero-argument forwards do not need that extra selection. Invalid fields
open their containing settings before the browser focuses them.

The normal form keeps the resolved setup and shows missing model, configuration
and input values directly. **Model creation → Python function** is visible in that form,
including its file and function when selected. **Advanced setup** contains
configuration-type overrides and explicit profile bindings. Switching the model creation
method preserves entered inputs and
restores the previous constructor or configuration values when switched back.

The factory can return the selected model component from an existing pipeline;
its modules can remain in their original files. For example, when opening the
`Decoder` class, a workspace `model_setup.py` can contain:

```python
from project.pipeline import build_pipeline

def make_decoder():
    return build_pipeline().decoder
```

Choose **Model creation → Python function**, set its file to `model_setup.py` and function to
`make_decoder`, then provide the decoder's example inputs. The returned component
must have the selected class identity. The factory executes real Python, including
its normal imports and constructor/checkpoint costs. This does not add capture of
arbitrary scripts or suppress imports.

The host owns the opening request, workspace writes, environment selection and
execution. The sidebar Canvas and browser present the same setup and progress.
Open as Graph uses the active Python editor when no class is supplied, with a class
picker for files containing several models. If no model ancestry can be proven,
explicit opening offers top-level subclasses as unverified choices; users select
one and the capture worker checks its actual module type. This does not change
the verified Models index. If a palette or view temporarily owns
focus, it uses the selected plain-text tab instead. It snapshots this URI before
asynchronous discovery and showing Canvas; an explicit CodeLens or Models target
always wins. Custom tabs, diff tabs and non-Python files never select an unrelated
recent model. With no selected Python file, the explicit workspace model/file
picker above is offered instead. Open Project remains a separate action.
MCP uses the same discovery and profile resolver. `get_graph_setup` reads progress
and choices; `suggest_graph_inputs` packages the unresolved values and a draft
skeleton for the connected agent's LLM; `configure_graph_setup` saves the completed
draft and continues the opening. Discovery reads Python syntax without importing
the project.
Complete, unambiguous source recipes can continue an explicit opening automatically.
Incomplete recipes prefill the form unless the bounded constructor-only preparation
can supply a supported input contract. Change inputs always opens the editable form.

Discovery recognizes imported dataclass configuration types, including aliases and
relative imports, and retains their declared fields and defaults. A narrow set of
known library decorators preserves readable declarations: Transformers documentation,
tuple-return, config-default and output-capture wrappers; Diffusers constructor
configuration registration; and Hugging Face strict dataclass configuration bindings.
Unknown decorators, shadowed imports and class-local replacements remain unresolved.
Recognition never imports or executes a decorator. Source fields alone do not prove
that a library model has valid conditional inputs or a complete configuration;
those models can still require setup. Automatic imported-config construction is
limited to supported plain dataclasses/attribute classes and simple configuration
constructors. The latter may assign local values and instance fields using supported
source expressions; a constructor parameter is never assumed to be an attribute
with the same name. Calls, branches, inheritance and other methods remain unresolved.

The discovery repair has regression coverage for each recognized form and unknown
or shadowed alternatives. The real worker check also covers a model with settings
in another source file: generated factory, TOML save/reopen, successful capture and
projection, and a deliberate forward failure that stays failed.

Source discovery also prepares a deterministic example recipe. It first reuses a
unique complete repository call, then concrete constructor/configuration values
and declared defaults. Self-contained inline calls inside expressions such as
`print(Model(...)(torch.ones(...)).shape)` retain their literal setup too.
It follows unchanged inputs through local methods and
nested modules to supported Linear, Conv1d/2d/3d, ConvTranspose1d/2d/3d,
Embedding, attention, Transformer and recurrent boundaries.
Known shape-preserving normalization/activation layers retain that relationship,
including Sequential prefixes, inline normalization and keyword calls. A Linear
boundary's flexible leading rank can agree with a stronger attention/conv shape;
incompatible dimensions still require setup.
An unresolved CLI expression such as `bilinear=args.bilinear` can use the declared
literal default for this example. The argument stays omitted so Python owns the
default, and the evidence identifies this as an example choice. A concrete source
call wins over this fallback; saved and user-supplied setup remain authoritative.
Required integer dimensions can receive a small synthetic value when their use in
a supported operator establishes the domain. This follows local nested constructors
and supported dataclass fields, independent of parameter/model names. Attention
width/head and convolution channel/group divisibility, supported integer arithmetic
and direct constructor comparisons constrain a joint choice. For example, an
unknown attention width with three known heads receives 9, not the usual example
width 8. A class count flowing to a Conv output dimension can receive an editable
example count; it is not the user's dataset class count. Source values and defaults
are preserved, including later positional-only arguments after a missing dimension.
Discovery never executes a constructor to make these choices. It considers at most
eight missing dimensions, values 1–256 and 4,096 assignments within its existing
syntax budget. Paths, callables, unsupported object types, nonliteral defaults,
unknown argument spreads and unsatisfied constraints still need setup. Discovered
factories retain their existing precedence instead of receiving invented arguments.
Batch size 1, CPU evaluation, seed 0 and any chosen spatial/token extent are
explicit example settings; layer widths, vocabulary bounds and declared image
sizes retain source evidence. Conflicting calls, transformed inputs, unknown
required values without supported constraints and incomplete source discovery remain unresolved. Binding analysis
only visits scopes that directly construct the selected class, so unrelated training
functions do not consume its evaluation budget. Unsupported constructor spreads
in a module-level invocation require explicit configuration. Incomplete factory
alternatives do not override the selected class’s usable defaults. Conditional or
augmented constructor assignments and Sequential subclasses modified after
initialization cannot establish fixed input dimensions. When direct construction
is unresolved and no concrete constructor/configuration values would be replaced,
a discovered factory callable without arguments can supply an example variant.
Public factories precede private helpers, then file and source order make the choice
deterministic. Conflicting source calls still require setup. Factory selection
discards dimensions inferred for the incomplete direct constructor. Discovery
never calls the factory.

When only inputs remain unresolved, with no supplied inputs or other unresolved
settings, the host may construct that selected model once in an isolated
CPU/evaluation worker. It can preserve bounded declared `dummy_inputs`, including
multi-input library examples, or infer one tensor from supported registered layers
and stored image-size metadata. A declared `main_input_name` can identify an
optional tensor input. This step does not call forward and provides no
measured graph evidence. It has a 30-second deadline, examines at most 4,096 modules
and limits the proposed tensor to 4,194,304 elements; it does not impose a model
allocation memory limit. Unsupported preparation returns to the setup form. Saved
or explicit inputs are never replaced. The prepared recipe still must be saved,
pass preflight and produce a real capture.
Import resolution is cached within each source snapshot; saving
a selected recipe checks its bindings without repeating automatic discovery. Discovery
never imports the model, reads arbitrary workspace files or tries candidate runs.
Automatic inputs total at most 4,194,304 elements. Undeclared image extents use 64
per spatial axis; undeclared token lengths use 8. These are labelled examples, not
proof that every downstream operation accepts them. `setup.py` is never executed;
package metadata alone does not establish a model configuration or input shape.

Direct `batch, sequence, width = x.shape` (or `x.size()`) establishes input rank
before a Linear boundary. Automatic examples preserve that rank and the layer's
last dimension, with labelled batch/leading extents; conflicting ranks remain
unresolved. The resolver does not infer ranks from unpacking transformed tensors.
For patch models without a declared image size, synthetic spatial extents are
rounded up to patch multiples (for example, 70 rather than 64 for patch size 14).
Declared image sizes and repository inputs remain unchanged. The existing total
element bound still applies. Real worker regressions cover rank-three sequence
inputs and patch-aligned images through profile persistence, capture/projection,
and deliberate forward failures.

Profiles live in `tensorviz.toml`. Automatic recipes are saved **before** execution
so source/configuration freshness checks cover the exact invocation. Failed recipes
remain editable; success never triggers another save or execution. Simple
construction uses explicit argument bindings; object construction uses a workspace factory. A profile records all
positional and keyword inputs, tensor dtypes and generators, mode and device.
Selected interpreters remain local workspace settings. Generated setup never
overwrites unrelated profiles or model source.

The setup service checks constructor and forward bindings together before returning
file contents. Generated profiles have a bounded marked section so later input
changes replace that section. Hand-authored profiles remain intact; editing one
through the form creates a new named profile. Invalid existing TOML blocks saving.
Configuration-object selections generate a content-addressed Python factory under
`.tensorviz/`; strings remain literal values. Neither discovery nor file preparation
imports the model or writes to the workspace.

The local session carries ordered `graphOpening` state and typed
`graphSetupRequest` actions. Only the owning host can publish opening state.
Setup actions identify the opening request and source target; supplied profile
data is bounded and revalidated by the host before any write or execution.
Snapshots can restore setup before the first graph has been projected. Resolution
drafts and provenance pass the same bounded checks in the host, browser and Python broker,
including reconnect snapshots. The shared protocol corpus covers these messages.

The host controller checks saved setup first. Invalid saved settings block automatic
resolution. With no matching profile, a complete source recipe is saved and then
passes the normal saved-profile preflight before one capture. An incomplete recipe
may first use the constructor-only input preparation described above; remaining
missing values require setup. Submitting the form resumes
the same opening request. Source changes, cancellation and retargeting invalidate
late discovery/save results; repeated form requests cannot start duplicate runs.
A failed forward may retain a captured prefix while the opening remains failed.

The selected profile name is remembered per workspace/model. Reopening with several
matching profiles reuses that choice. A removed choice blocks instead of switching
to another invocation. An explicit empty input binding supports zero-argument
forwards; signature preflight still rejects missing required arguments.

For direct wrappers such as DINOv2's `forward(*args, is_training=False, **kwargs)`,
discovery follows an unconditional first call to a method declared on the same
class. Forwarded arguments appear in setup (`x`, optional `masks`) alongside the
wrapper's controls. The shared profile check rejects missing `x`, including in an
existing saved profile, before launching the worker. It does not select dimensions.
Conditional, decorated, reassigned or transformed forwarding remains unresolved;
those models need explicit bindings in Advanced setup.

Setup persistence checks the original config bytes and file identity before one
VS Code workspace edit. The host saves the prepared files, then verifies their
contents before allowing capture to resume. Dirty files, changed destinations,
linked config files and linked helper directories refuse the write. Generated
factory filenames include a content digest; an edited existing helper is preserved.

Opening requires Workspace Trust and saved execution sources. Dirty sources offer
Save & Open. Setup completion resumes opening; it does not require a separate
Validate click. Cancellation, retargeting and source changes invalidate pending
work. Reconnection restores the current opening state.

The normal path never searches candidate shapes. A failed forward keeps the
selected setup, reopens its fields automatically, and shows its captured prefix
and failure. A successful worker outcome without a projected capture is a failed
opening, never a ready AST graph. Missing dependencies
offer environment repair; missing constructor values offer setup. Source-only
inspection remains an explicit action. Refresh graph and Change inputs remain
available after opening.
Legacy validation requests also require a matching profile. Missing or unrelated
profiles return setup guidance without launching an automatic input search.

## Unchanged calls

The canvas hides captured leaf `Identity` and `Dropout` calls when a complete
capture proves that their sole tensor input and output are the same value and
they recorded no operations. Custom modules, failed calls, partial captures and
the root model remain visible. Structured inputs or outputs also remain visible;
matching one nested tensor alone cannot prove the whole interface unchanged.
Every call stays in the canonical graph document.

**More tools → Show unchanged calls (N)** reveals them inside expanded modules.
The offline viewer keeps its own visibility button. Each compact
card shows its actual module slot and type, such as `ls1 · Identity`, its full
path, and **Input unchanged**. When the projection has a source mapping, the title
opens it in VS Code. Otherwise its tooltip says **Source location not mapped**;
selecting the call still exposes its captured details.
A dashed connector identifies the recorded input; it is a display annotation and
does not change tensor producers or computation edges. Empty captured calls use
**No recorded operations** instead of an unavailable-contents warning.

Quick jump accepts captured paths and reveals the requested call. MCP focus,
offline search and walkthrough targets also reveal hidden calls. Sidebar
hide/show retains visibility and rebuilds the annotations. Export retains all
calls, ports and canonical edges, and excludes the display annotations. The
offline viewer derives folding again from the validated capture.

The workspace adapter discovers against live source digests, rechecks all supplied
files before saving, and continues capture with the resolved profile. Dirty Python
files offer Save & Open; environment repair and config editing return to opening.

Both views keep the initial source projection behind setup/progress. The canvas
stays mounted, and setup uses a floating panel in its existing theme. After capture,
the panel becomes a compact input/profile row above the canvas; **Edit inputs**
expands it. The row reserves space so search and overlay controls stay accessible.
Changing inputs or capturing again keeps the previous captured graph visible and
labelled. A ready message cannot expose a static or mismatched projection as a
captured graph. A workspace switch clears the old projection even for matching
file and class names. Refresh graph remains in the canvas toolbar. Constructor
defaults remain omitted unless edited. Unambiguous examples can open automatically;
conflicting examples require selection.
Factory scaffolds open as unsaved Python in VS Code and preserve existing helpers.
Hiding Canvas preserves setup and capture. Recreating either view restores the
opening target even before a projection exists. Environment recovery restarts
capture for the retained model without opening another surface.
Switching construction modes removes unused fields before wire validation.
Starting another opening clears the previous verdict toast. Completed attempts
show one toast per run; the latest result remains available beside Refresh graph.
Captured projections use the source files recorded by that invocation. Unrelated
workspace files, including another model's generated factory, do not block display.
Source digests for the captured files must still match.
VS Code may release clean documents while another model runs. Losing an in-memory
document revision alone does not invalidate saved execution; changed saved bytes,
physical file identities, dirty buffers and revisions of still-loaded documents do.


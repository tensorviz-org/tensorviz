# Saved model validation profiles

## Contents

- [Example: two inputs for one model](#example-two-inputs-for-one-model)
- [Execution environment](#execution-environment)
- [Models constructed from Python objects](#models-constructed-from-python-objects)
- [When validation cannot start](#when-validation-cannot-start)
- [Python example inputs](#python-example-inputs)
- [Large graph capture budgets](#large-graph-capture-budgets)

**Open as Graph** captures the saved class with the selected Python environment.
A project-root `tensorviz.toml` pins constructor arguments, input tensors, device,
mode and timeout. Source discovery does not import the project. Missing setup is
collected in the browser, saved by VS Code, then captured automatically.

Model profiles may set `framework = "jax"` for plain JAX functions and Flax
modules. Omitting it retains PyTorch behavior. JAX traces infer shapes on the saved CPU or CUDA device; see [JAX and Flax graphs](jax-support.md) for setup,
factories and the supported evidence.

Execution setup failures identify their stage and, when available, missing or
unexpected Python parameter names. Binding errors are checked before forward
execution. Missing imports, unavailable devices, invalid object factories, missing
fixture/checkpoint files and input materialization failures refuse the requested
setup without substituting defaults. The browser gives a bounded explanation;
MCP retains the typed diagnostic. Detailed Python exceptions stay in VS Code Output.

For a failed forward, supported matrix and convolution operators report only the
dimension constraints established by their actual operands. Unknown dimensions are
shown as `?`. Matrix and embedding failures can also report incompatible dtypes.
Assertions and unsupported operators remain generic failures; exception text is
never used to invent an expected shape. A partial graph is evidence from that input,
not a successful validation or proof about other branches.

Without a matching saved profile, opening resolves one deterministic example from
an explicit numeric input declaration, an unambiguous repository call or supported
source/constructed-model constraints. Complete setup is saved before capture;
unresolved or conflicting choices open the form. Dimensions are not searched.
One matching profile runs directly. Multiple matches use the native picker once;
subsequent openings reuse the remembered choice. A removed choice, invalid profile
or cancelled picker never substitutes another invocation. Change inputs edits the
selected setup. Generated profile sections can be updated; hand-written sections
are preserved and form edits create a new profile.

### Numeric input declarations

For initial setup, a Python comment such as `# tensorviz: input=5,8` supplies
positional float32 tensors with normal values and seed 0.
`# tensorviz: input=2,8;2,8` supplies two tensors in that order. Dimensions must
be positive, with at most 32 dimensions per tensor, 64 tensors and 4,194,304 elements.

Place the comment inside the model class or directly above its decorators/class
line, with no blank line or code between. It belongs only to that class. A comment
elsewhere applies only when the file has one model class; in a file with several
models, move it beside the intended class or use Inputs/TOML. Generated templates
place it directly above the model. Source previews and capture use the same class
ownership rules. Declaration text inside a string is ignored. Carriage-return line
endings and UTF-8 files with a byte order mark retain their declarations.

Conflicting declarations, unsupported expressions and incompatible argument counts
require explicit setup. Comment expressions are not evaluated. Use Inputs/TOML
for other dtypes, keyword arguments and nested data. A saved profile remains
authoritative; editing the comment does not replace its invocation.

## Example: two inputs for one model

For `GoodNet` in `audit_models.py`, with an empty constructor and a forward input
of shape `[batch, 4]`, save this as `tensorviz.toml` in the workspace root:

```toml
version = 1

[profiles.audit_small]
kind = "model"
model = "audit_models:GoodNet"
mode = "eval"
device = "cpu"
grad_enabled = false
limits = { timeout_ms = 30000 }
[profiles.audit_small.init]
args = []
kwargs = {}
[profiles.audit_small.inputs]
args = [{kind="tensor", shape=[2,4], dtype="float32", requires_grad=false, generator={kind="full", value=1}}]
kwargs = {}

[profiles.audit_large]
kind = "model"
model = "audit_models:GoodNet"
mode = "eval"
device = "cpu"
grad_enabled = false
limits = { timeout_ms = 30000 }
[profiles.audit_large.init]
args = []
kwargs = {}
[profiles.audit_large.inputs]
args = [{kind="tensor", shape=[3,4], dtype="float32", requires_grad=false, generator={kind="full", value=1}}]
kwargs = {}
```

The model fixture `backend/tests/fixtures/model_profiles/audit_models.py`
maps these inputs to `[2,2]` and `[3,2]` outputs respectively. The
[valid TOML fixture](../backend/tests/fixtures/model_profiles/valid.toml)
matches this example. For a package, use `model = "package.module:ClassName"`;
the corresponding file or package `__init__.py` must match the class on the canvas.
`cwd`, if provided, is relative to the workspace root.

The worker honors `cwd` during construction and the forward, as well as
`grad_enabled` and `deterministic` (default `true`). Omitting the model `seed` leaves
global RNGs unseeded by tensorViz; an explicit nonnegative safe integer seeds Python,
Torch and NumPy (NumPy uses the low 32 bits). Per-tensor generator seeds remain
independent and are always materialized as written. A forward with gradients enabled
does not perform a backward pass or optimizer update.

`limits.timeout_ms` is an integer number of milliseconds from **100 through
86,400,000** inclusive. Omitting it uses the profile default of **60,000 ms**.
It belongs inside `limits`, either as the inline table above or under
`[profiles.audit_small.limits]`. A bare `timeout_ms` at the profile root is invalid.
Automatic validation without an applicable profile retains its separate
**180,000 ms** forward limit.

## Execution environment

Develop and run the tool service with the Python in the [versions table](local-setup.md#versions). Model capture uses the
interpreter you select. Expression-level source edits need source positions that
Python 3.10 does not record; capture, tensor shapes, and line-level source
evidence do not. The capture profile in `backend/requirements-capture.txt`
includes Torch, Pydantic, packaging, and NumPy, and conditional `tomli` where
the interpreter has no `tomllib`. It does not include the web-server libraries.

Saved-profile lookup, browser/MCP capture and candidate execution use the same
read-only interpreter resolver. A remembered project or
managed choice wins over automatic discovery. A selected managed runtime must
already be ready; these reads never download or install it. With no remembered
choice, an explicit `tensorviz.pythonPath` takes precedence over the Python
extension, then an existing managed runtime can be used if no project interpreter
is available. An explicit missing path or unavailable selected runtime refuses
execution instead of silently changing environments. Once resolved, the saved
plan pins the interpreter and rechecks it before and after execution.

## Models constructed from Python objects

When the constructor needs an object such as Whisper's `ModelDimensions`, use a
saved workspace factory instead of trying to put the object into JSON/TOML. Keep
`model` pointing to the class shown on the graph. For example, save a small factory
in `recipes.py`:

```python
from whisper.model import ModelDimensions, Whisper

def make_model():
    return Whisper(ModelDimensions(
        n_mels=80, n_audio_ctx=16, n_audio_state=32, n_audio_head=4,
        n_audio_layer=2, n_vocab=128, n_text_ctx=16, n_text_state=32,
        n_text_head=4, n_text_layer=2,
    ))
```

Add this profile to the workspace-root config (with `version = 1` once at its root):

```toml
[profiles.whisper]
kind = "model"
model = "whisper.model:Whisper"
mode = "eval"
device = "cpu"
grad_enabled = false
seed = 0
factory = { file = "recipes.py", callable = "make_model", kwargs = {} }
[profiles.whisper.inputs]
args = [
  {kind="tensor", shape=[1,80,32], dtype="float32", requires_grad=false, generator={kind="full", value=1}},
  {kind="tensor", shape=[1,4], dtype="int64", requires_grad=false, generator={kind="full", value=1}},
]
kwargs = {}
```

The `configured` profiles in
[`examples/tensorviz-demo/tensorviz.toml`](../examples/tensorviz-demo/tensorviz.toml)
and [`examples/tensorviz-sweep/tensorviz.toml`](../examples/tensorviz-sweep/tensorviz.toml)
use this mechanism for local config dataclasses. Open each example folder as the
workspace root; its factory paths are relative to that folder.

Choose exactly one of `init` or `factory`, including when `init` would be empty.
Factory `file` is workspace-root-relative (independent of profile `cwd`); `callable`
names a top-level callable and `kwargs` contains bounded JSON literals. The factory
must return the exact selected class, not a subclass or another class with the same
name. It runs once inside the approved worker using the selected Python environment.
No expression evaluator or automatic dependency installation is involved.

Preflight parses the factory's signature and tracks its source without importing it.
Missing, malformed, escaped or unsaved factory files refuse execution. Imported
helpers remain subject to the existing saved-source checks. Browser Validate and
explicit MCP setup use the same factory loader and class-identity check. Script
profiles remain unavailable through the browser's model-validation control.

## When validation cannot start

An unclassified construction failure points to the explicit model profile and
the underlying exception in tensorViz Output. It does not assume that constructor
overrides can fix an unknown error; the construction-advice regression checks
preserve that distinction.

Malformed TOML, unknown keys, invalid values, unsaved configuration, a failed
snapshot, or an unresolved/unsupported selected profile produce **Could not run**
with bounded, actionable feedback. Fix/save the named fields or check the selected
Python environment, then retry. For a misplaced timeout, feedback names
`profiles.<name>.timeout_ms` and directs you to `limits`. The canonical config is
validated as a whole: malformed profiles are not silently discarded, including
malformed profiles for another class. Script profiles cannot run through this
model-validation control.

Automatic validation requires a known positional forward signature. Variadic,
unavailable or required keyword-only signatures, and more than three required
positional inputs, request an explicit saved invocation. Unknown arity is never
treated as zero inputs. Explicit bindings can still run variadic and keyword-only
models. If an automatic candidate produced only a failed prefix, its input is
labelled **failed guess**, not described as having completed a successful forward.

No automatic input or alternate profile substitutes for failed explicit settings.
Directories, unreadable files, broken symlinks and snapshot errors are not treated
as absence. Selection is resolved again after the picker; removal or retargeting
of the selected profile refuses. Cancellation wins over late snapshot/picker
responses and releases the request so another attempt can start.

The resolved profile retains the identities of the configuration, selected Python
environment, tracked native files and conservative workspace source set. The host
rechecks them after interpreter probing, after capture and after graph refresh.
Changed settings refuse certification; an environment resolved for one profile
cannot silently switch to another interpreter. Opening an unchanged saved file is
allowed, while changed observed document versions or physical file identities refuse.
A settings change racing graph refresh marks the retained graph stale.

Factories can import workspace packages, including a package named `services`:
planned graph and validation workers keep their own implementation in private
module namespaces. Both apply the declared seed/device policy before importing
the model, so import-time random initialization follows the same execution context.

Refusal and cancellation do not certify the model. Existing graph measurements
remain available and, when no new measured projection arrived, the page labels
them **Previous measurements retained**. Neither outcome triggers a success sweep.
Saving/editing source retains the existing source-freshness and stale-evidence rules.

## Python example inputs

A model profile can reuse an explicit workspace function instead of duplicating
its tensors in TOML. The input factory is separate from the model constructor:

```toml
inputs = { factory = { file = "model.py", callable = "graph_inputs", kwargs = {} } }
```

Return a dictionary of named forward inputs, or exactly `{ "args": (...),
"kwargs": {...} }` for positional arguments. The function runs once per capture
inside the existing worker, after construction, with the selected runtime/device
and timeout. Preflight checks its saved source/signature without importing it;
its file identity participates in freshness checks. Normal explicit bindings
remain supported and cannot be mixed with a factory. The setup form's **Use
Python example inputs** control switches between these choices without losing
in-progress tensor settings. Function keyword options remain editable in Advanced
setup.

Returned data supports the existing nested tensor/scalar fixture types, bounded
to 4,194,304 total tensor elements, 256 tensors, 64 MiB of tensor data and the
existing depth/entry limits. These are checked after the function returns; Python
execution uses the existing worker timeout, not an OS memory sandbox. The 4,096
literal-value limit is unchanged. A factory preserves exact values/dtypes and
repeated tensor references; it does not guess model inputs.

## Large graph capture budgets

Canonical graph captures default to a 16 MiB recording budget, independently of
the 4 MiB validation-record limit. Model profiles may set
`limits.max_capture_bytes` from 1 KiB through 16 MiB to choose a smaller recording
budget. Explicit limits remain authoritative. `limits.max_record_bytes` bounds
validation records; raising it does not expand the graph recording budget.

Recording limits apply to captured evidence, not parameter storage: a FakeTensor
model can have billions of parameter elements without allocating their values.
A successful forward still needs its coverage checked. A `byte_limit` omission
means the recording is partial even when the forward passed; inspect the resolved
`captureLimits.bytes` before changing the recipe.

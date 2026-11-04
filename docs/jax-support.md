# JAX and Flax graphs

TensorViz opens plain JAX functions and Flax Linen/NNX modules through the same
Open as Graph, saved setup, Canvas and browser workflow as PyTorch. JAX and Flax
are optional model dependencies in the selected capture Python. Missing libraries
use the existing reviewed dependency installation flow. The shared capture worker
still requires its capture dependencies, including Torch; adding JAX does not
replace the selected Torch/CUDA build.

JAX graphs describe a **traced program** from `jax.make_jaxpr`, for the saved input
shapes, dtypes and static Python values. Shape evidence is **inferred**, not a
measurement of an eager forward. Tracing runs Python and model setup under the
existing Workspace Trust, saved-source, selected-interpreter and worker deadline
checks. Source-only discovery never imports the model or JAX.

## Open a function

Save this as `model.py`, choose **Open as Graph** on `net`, then set `x` to a
float32 tensor with shape `2, 3` in the input form:

```python
import jax
import jax.numpy as jnp

@jax.jit
def net(x):
    return jax.nn.relu(x @ jnp.ones((3, 4)))
```

The equivalent saved recipe is:

```toml
version = 1

[profiles.jax_example]
kind = "model"
framework = "jax"
model = "model:net"
mode = "eval"
device = "cpu"
grad_enabled = false
seed = 0
init = { args = [], kwargs = {} }
inputs = { args = [{ kind = "tensor", shape = [2, 3], dtype = "float32", requires_grad = false, generator = { kind = "full", value = 1 } }], kwargs = {} }
```

Omitting `framework` retains existing PyTorch behavior. Plain JAX functions take
their arguments from `inputs`, with an empty `init`. Array leaves in registered JAX PyTrees are traced; Python scalar arguments remain static.
Lists, tuples and dictionaries retain their structure. Flax structs, named tuples,
FrozenDicts and other registered containers normalize to dictionaries of their
registered child keys. Static PyTree metadata is not serialized. Capture never
reconstructs these containers with graph-reference objects, so array invariants
in a custom constructor remain valid.
This permits an explicitly configured boolean to select Python control flow.
Changing any such value requires another trace.

Discovery recognizes top-level functions using imported JAX operations or
decorators, and classes deriving from the documented `flax.linen.Module` and
`flax.nnx.Module` bases, including local inherited/re-exported bases. It does not
classify every function in a file merely because that file imports JAX.

## Open Flax

```python
from flax import linen as nn

class MLP(nn.Module):
    width: int = 8

    @nn.compact
    def __call__(self, x):
        return nn.Dense(4)(nn.relu(nn.Dense(self.width)(x)))
```

Open `MLP`, enter the example input, and retain `framework = "jax"` in its recipe.
Linen constructor annotations populate editable setup fields. An unbound Linen
module is initialized with `jax.random.key(seed)` and the saved inputs. This is
freshly initialized state, not trained weights. Its `__call__` is then traced.

For saved variables, replace `init` with a workspace factory returning the exact
selected Linen class bound with `model.bind(variables)`. NNX modules can use the
same factory mechanism to construct explicit `nnx.Rngs`, layers and state. A
factory for a plain function may return a configured callable. Factories run only
inside the approved worker; their files participate in source freshness checks.

Python input functions use the existing `inputs.factory` contract and can return
`{"args": (...,), "kwargs": {...}}` or a mapping of named inputs. Use these for
parameter PyTrees, typed PRNG keys, or loading JAX-specific fixtures. See
[Python example inputs](model-validation-profiles.md#python-example-inputs).

## Graph evidence and limits

- Primitive names, argument positions, repeated uses, value shapes/dtypes and
  nested output structure are retained in the canonical capture. JIT, checkpoint (`remat2`) and
  unconditional custom-derivative call bodies expand without executing kernels.
  These groups describe tracing scopes, not a recovered Flax layer hierarchy.
- `cond`, `scan`, `while` and other unexpanded subprograms retain their primitive
  boundary and output shapes. Capture coverage is partial with `opaque_kernel`;
  neither a branch nor an iteration is reported as executed. Effects are also
  explicitly incomplete. Convolution/gather dimension tuples, precision and gather-mode enums, and
  ragged-dot dimension metadata are retained. Other unsupported metadata stays opaque.
  Scatter reducer subprograms remain explicit opaque boundaries, including in MoE models.
- JAX source frames provide navigation only. PyTorch source-edit recipes,
  training-step capture, storage aliases, activation statistics and
  output-byte measurements are unavailable for Jaxpr captures. Large graphs can use Fit below the default
  minimum zoom; Readable detail returns to legible node cards.
- Setup uses the saved `cpu` or `cuda:N` device, `eval`, and `grad_enabled = false`.
  The selected Python must have a compatible JAX CUDA installation for NVIDIA GPUs;
  an unavailable backend or index fails before model import, without CPU fallback.
  The device scopes source import, initialization, input factories and tracing.
  Explicitly placed or sharded arrays from a factory retain their own placement,
  following JAX's `default_device` semantics. Graph values remain abstract and do
  not claim observed GPU execution. Express `grad`,
  `vmap`, or other transformations in the selected callable. JAX has no per-array
  `requires_grad` flag. Input descriptions requesting one fail explicitly.
- 64-bit input descriptions require `JAX_ENABLE_X64=1` in the capture environment;
  TensorViz refuses silent dtype narrowing. No Torch checkpoint/fixture loader is
  used for JAX: use an explicit workspace factory instead.
- Existing operation, call, value, byte, timeout and source-freshness limits apply.
  Explicit inputs are bounded to 256 arrays and 4,194,304 total elements. Unknown
  input dimensions are requested rather than guessed. Failed tracing publishes
  a failure, never a successful empty graph.

NNX modules, bound methods and partial callables clone their state wrappers inside
the JAX trace context, allowing `nnx.jit` to update temporary state without crossing
trace levels. Underlying arrays are shared rather than duplicated. Arbitrary Python
closures still follow JAX/NNX trace rules.

The capture records `backend = "jaxpr"`, its JAX version and
`coverage.scope = "traced_program"`. It cannot carry runtime timings or claim
PyTorch execution coverage. Shared graph projection, paginated inspection and
portable graph files retain this distinction.

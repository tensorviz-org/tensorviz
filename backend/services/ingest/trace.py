"""Execution tracing: real tensor shapes and true dataflow, with no weights.

Method selection was decided by measurement, not preference:

* ``torch.fx.symbolic_trace`` failed on 8 of 8 real models (``co_varnames is too small``,
  ``Proxy`` item assignment, control-flow ``TraceError``). Not used.
* ``torch.export.export`` survived 5 of 8; it is the only route to *symbolic* dims but
  cannot be the primary path.
* forward hooks under ``FakeTensorMode`` survived 14 of 16 architectures at 24-289 ms and
  yield real in/out shapes. This is the primary path.

Module hooks alone are not enough for EDGES. The tensor leaving module A usually passes
through a functional op (``x + residual``, ``reshape``, ``cat``) before reaching B; those
ops are not modules, so a hook never sees them and the A->B edge is lost (measured: gpt2
yielded 4 edges that way). ``TorchDispatchMode`` intercepts every aten op and a
hook-maintained module stack attributes each one to its enclosing module (100%
attribution measured). Producer/consumer then follows from tensor identity — which is
what recovers a UNet's long-range skips (``down_blocks.0 -> up_blocks.3``).

Hooks are registered with ``with_kwargs=True``: a UNet passes its skip tensors as a kwarg
bundle (``res_hidden_states_tuple``) that a positional-only hook cannot see.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from ..semantic.identity import weak_identity

# aten calls that carry no dataflow meaning. ``prim.device`` alone accounted for 736 of
# 1621 ops on gpt2, so filtering is a real reduction, not a micro-optimisation.
NOISE_OPS = (
    "prim.device", "aten.sym_size", "aten._to_copy", "aten.empty_like",
    "aten.detach", "aten.lift_fresh", "aten.alias", "aten.sym_numel",
)
# A FakeTensor's ``.device`` is itself a dispatched ``prim.device`` call (a real
# tensor's is not), so the canonical recorder skips it as the legacy fold does.
_PRIM_DEVICE = torch.ops.prim.device.default

# Exceptions raised when a model's control flow needs real tensor VALUES, which a fake
# tensor cannot supply. Mixture-of-Experts routing (`idx.tolist()`), switch_transformers
# (`aten.nonzero`) and longformer (`.item()`) all land here.
try:  # pragma: no cover - availability differs across torch builds
    from torch._subclasses.fake_tensor import (
        DataDependentOutputException,
        DynamicOutputShapeException,
    )
    DATA_DEPENDENT = (DataDependentOutputException, DynamicOutputShapeException)
except Exception:  # pragma: no cover
    DATA_DEPENDENT = ()


def _tensors(value: Any, acc: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
    acc = [] if acc is None else acc
    if isinstance(value, torch.Tensor):
        acc.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _tensors(item, acc)
    elif isinstance(value, dict):
        for item in value.values():
            _tensors(item, acc)
    elif hasattr(value, "to_tuple"):  # HF ModelOutput
        try:
            _tensors(value.to_tuple(), acc)
        except Exception:
            pass
    return acc


def _first_shape(value: Any) -> Optional[List[int]]:
    found = _tensors(value)
    return [int(d) for d in found[0].shape] if found else None


def aten_base(op_name: str) -> str:
    """``'aten.add_.Tensor'`` -> ``'add'``. Empty string for anything not an aten op.

    The trailing-underscore strip matters: a residual is written ``out += identity``,
    which dispatches as the in-place ``aten.add_``. Splitting on ``.`` and taking one
    component also keeps ``aten.addmm`` from being mistaken for an ``add``.
    """
    if not op_name.startswith("aten."):
        return ""
    parts = op_name.split(".")
    return parts[1].rstrip("_") if len(parts) > 1 else ""


# Functional ops that may earn a node. Kept here because __torch_dispatch__ needs it to
# decide whether recording shapes is worth the allocation; the POLICY of what to do with
# them lives in materialise.py.
_SHAPE_WORTHY = {
    "add", "mul", "cat", "concat", "view", "reshape", "flatten", "_unsafe_view",
    # Functional pooling. MobileNetV2 calls nn.functional.adaptive_avg_pool2d rather than
    # holding a module, and it lowers to aten.mean.dim — so without a node the classifier
    # received [1, 1280, 7, 7] flattened to 1x62720 instead of 1x1280.
    "mean", "adaptive_avg_pool2d", "_adaptive_avg_pool2d", "avg_pool2d", "max_pool2d",
}


class _Provenance(TorchDispatchMode):
    """Records every aten op with the module it ran inside and its tensor identities.

    Each op also carries the *invocation* of its owner (``owner_call``), not just the
    owner's name. torchvision's BasicBlock calls ``self.relu`` twice, and with only a
    name both calls collapse onto one node — which is what gave layer1.0.relu three
    inbound edges and made the imported graph CYCLIC
    (relu -> conv2 -> bn2 -> relu). The counter is per enclosing invocation, so a
    module shared across parents (ALBERT) still reports call 1 each time.
    """

    def __init__(self, frame_roots: Sequence[str] = ()) -> None:
        super().__init__()
        self.ops: List[Dict[str, Any]] = []
        self.stack: List[str] = []
        self.calls: List[int] = []
        # Directories whose Python frames count as "the user's code". Empty means
        # capture nothing, which is the pre-M4 behaviour and the flag-off state
        # scripts/bench_frame_capture.py exists to justify.
        self.frame_roots: Tuple[str, ...] = tuple(
            os.path.realpath(root).rstrip(os.sep) + os.sep for root in frame_roots if root
        )
        self.frames_captured = 0
        # co_filename -> its resolved path if it is a user frame, else "". The walk
        # below runs for every recorded op, and resolving each frame's file with
        # realpath (several lstat calls) was most of the trace time on larger
        # models. A file's resolution does not change within a single-use worker;
        # a relative co_filename is keyed with the cwd it was resolved against.
        self._frame_files: Dict[Any, str] = {}
        self._own_directory = os.path.realpath(os.path.dirname(__file__)) + os.sep
        self.evidence: Any = None
        self.semantic: Any = None
        self.failed_operation: Optional[Tuple[BaseException, Dict[str, Any]]] = None
        # Where the ROOT module's forward began and ended, as indices into `ops`.
        # Only a traced call that does more than a forward needs them (task.md M9's
        # `--kind train_step` runs a backward and an optimiser step inside the same
        # dispatch mode): gradient accumulation is a chain of `add`s with two known
        # producers each, which `resolve_dataflow`'s merge rule would materialise as
        # eight nodes the model does not have. `None` while nothing has entered.
        self.forward_start: Optional[int] = None
        self.forward_end: Optional[int] = None
        # One counter dict per open invocation: _frames[-1] counts how many times each
        # child has been entered during the CURRENT invocation of the enclosing module.
        self._frames: List[Dict[str, int]] = [{}]
        # Sets of names reused more than once inside a single enclosing invocation.
        self.reuse: List[frozenset] = []
        # Monotonic identities survive address reuse without keeping activations alive.
        self._tensor_objects: Dict[int, Any] = {}
        self._identity_counts: Dict[str, int] = {}

    def tensor_id(self, tensor: torch.Tensor) -> str:
        return weak_identity(self._tensor_objects, self._identity_counts, tensor, "tensor:")

    def push(self, name: str) -> int:
        frame = self._frames[-1]
        frame[name] = frame.get(name, 0) + 1
        index = frame[name]
        self.stack.append(name)
        self.calls.append(index)
        self._frames.append({})
        return index

    def pop(self) -> None:
        if self._frames:
            closing = self._frames.pop()
            repeated = frozenset(n for n, c in closing.items() if c > 1)
            if repeated and self.semantic is None:
                self.reuse.append(repeated)
        if self.stack:
            self.stack.pop()
        if self.calls:
            self.calls.pop()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: D105
        if self.semantic is None:
            return self._legacy_dispatch(func, args, kwargs or {})
        semantic = self.semantic
        if func is _PRIM_DEVICE:
            return func(*args, **(kwargs or {}))
        record = semantic.guard(semantic.begin, func, args, kwargs or {})
        # Bounded diagnostic digests are instrumentation, not user tensor accesses.
        # Canonical runs must not also accumulate the unbounded legacy op stream.
        prior = semantic.busy
        semantic.busy = True
        try:
            output = self._execute_observed(func, args, kwargs or {})
        except BaseException as error:
            semantic.guard(semantic.failed, record, func, error, self.failed_operation)
            raise
        finally:
            semantic.busy = prior
        semantic.guard(semantic.succeeded, record, func, args, kwargs or {}, output)
        return output

    def _execute_observed(self, func, args, kwargs):
        kwargs = kwargs or {}
        name = str(func)
        observed = self.evidence.begin(name, self.stack[-1] if self.stack else "",
            self.calls[-1] if self.calls else 1, args, kwargs, self._source_frame()) if self.evidence else None
        try:
            out = func(*args, **kwargs)
        except Exception as exc:
            from ..semantic.diagnostics import operation_diagnostic
            self.failed_operation = (exc, operation_diagnostic(name, args, kwargs, exc))
            if observed is not None:
                observed['diagnostic'] = self.failed_operation[1]
                self.evidence.fail(observed, exc)
            raise
        if observed is not None:
            self.evidence.finish(observed, out)
        return out

    def _legacy_dispatch(self, func, args, kwargs):
        out = self._execute_observed(func, args, kwargs)
        name = str(func)
        if any(noise in name for noise in NOISE_OPS):
            return out
        ins = _tensors(args) + _tensors(kwargs)
        outs = _tensors(out)
        base = aten_base(name)
        record: Dict[str, Any] = {
            "op": name,
            "owner": self.stack[-1] if self.stack else "",
            "owner_call": self.calls[-1] if self.calls else 1,
            "fn": base,
            "in_ids": [self.tensor_id(t) for t in ins],
            "out_ids": [self.tensor_id(t) for t in outs],
        }
        if base in _SHAPE_WORTHY:
            record["in_shape"] = [int(d) for d in ins[0].shape] if ins else None
            record["out_shape"] = [int(d) for d in outs[0].shape] if outs else None
            dim = kwargs.get("dim")
            if isinstance(dim, int):
                record["dim"] = dim
        if self.frame_roots:
            frame = self._source_frame()
            if frame is not None:
                # `frame` stays a 2-tuple: services/ingest/materialise.py accepts one
                # only when `len(frame) == 2`, so widening it here would silently strip
                # the source line off every materialised merge. The enclosing function
                # rides alongside instead (task.md M9 needs it to name a frame node).
                record["frame"] = (frame[0], frame[1])
                record["frame_fn"] = frame[2]
                self.frames_captured += 1
        self.ops.append(record)
        return out

    def _source_frame(self) -> Optional[Tuple[str, int, str]]:
        """The innermost live frame belonging to the user's code, and its function.

        Innermost, not outermost: ``x + self.l2(h)`` runs its add directly in
        ``forward``, while ``self.l1(x)`` runs its addmm several torch frames
        deeper — and both must answer with the ``forward`` line the reader is
        looking at, not with ``nn/modules/module.py``. Walking up from the
        dispatch and stopping at the first frame under a root does exactly that,
        and skips this module's own frames for free (trace.py is not under a
        user root).

        The cost of this walk is the whole reason ``frame_roots`` is a flag: it
        runs once per recorded aten op, and a gpt2 forward records hundreds.
        """
        frame: Any = sys._getframe()
        files = self._frame_files
        while frame is not None:
            code = frame.f_code
            filename = code.co_filename
            # co_filename is whatever the import machinery recorded, which is
            # relative when the module was found through a relative sys.path
            # entry. Compare like with like or every frame misses.
            absolute = os.path.isabs(filename)
            key = filename if absolute else (filename, os.getcwd())
            resolved = files.get(key)
            if resolved is None:
                resolved = os.path.realpath(filename if absolute else os.path.abspath(filename))
                if not resolved.startswith(self.frame_roots) or resolved.startswith(self._own_directory):
                    resolved = ""
                files[key] = resolved
            if resolved:
                return (resolved, frame.f_lineno, code.co_name)
            frame = frame.f_back
        return None


class TraceResult:
    def __init__(self) -> None:
        self.io: Dict[str, Dict[str, Any]] = {}
        self.call_counts: Dict[str, int] = {}
        self.ops: List[Dict[str, Any]] = []
        self.input_ids: List[str] = []
        self.visited: Set[str] = set()
        # Names reused more than once inside one invocation of their enclosing module.
        self.reuse_scopes: List[frozenset] = []
        self.tier: str = "tree-only"
        self.error: Optional[str] = None
        # The Python frames the failure came through, innermost last. The one thing a
        # message like "mat1 and mat2 shapes cannot be multiplied" cannot tell you is
        # WHICH line, and that is the whole content of the editor diagnostic.
        self.error_frames: List[Tuple[str, int, str]] = []
        self.elapsed_ms: float = 0.0
        self.frames_captured: int = 0
        self.evidence: Optional[Dict[str, Any]] = None
        self.diagnostic: Optional[Dict[str, Any]] = None
        self.semantic_capture: Optional[Dict[str, Any]] = None
        self.semantic_error: Optional[str] = None


def _install_hooks(model: nn.Module, prov: _Provenance, result: TraceResult) -> List[Any]:
    """Scope complete eager calls, including user hooks and caught exceptions.

    Always-called forward hooks can run even when a global pre-hook fails before
    our local pre-hook. Wrapping the eager call pairs recursive invocations with
    try/finally, preserving user hooks and successful None outputs.
    """
    from types import SimpleNamespace
    handles: List[Any] = []
    missing = object()

    def install(name: str, mod: nn.Module) -> None:
        original = mod._call_impl
        own = mod.__dict__.get('_call_impl', missing)

        def call(*args, **kwargs):
            if prov.semantic is not None and threading.get_ident() != prov.semantic.thread:
                prov.semantic.omit("unsupported_type")
                return original(*args, **kwargs)
            if name == "" and prov.forward_start is None:
                prov.forward_start = len(prov.ops)
            prov.push(name)
            if prov.evidence is not None:
                prov.evidence.enter(name, args, kwargs)
            result.visited.add(name)
            result.call_counts[name] = result.call_counts.get(name, 0) + 1
            rec = result.io.setdefault(name, {"in": None, "out": None})
            # A later failed/reused call cannot borrow an earlier output's bytes.
            rec["out"] = None
            rec.pop("output_tensor", None)
            if rec["in"] is None:
                rec["in"] = _first_shape(args) or _first_shape(kwargs)
            if prov.semantic is not None:
                prov.semantic.guard(prov.semantic.enter, mod, args, kwargs)
            try:
                output = original(*args, **kwargs)
            except BaseException as error:
                if prov.semantic is not None:
                    prov.semantic.guard(prov.semantic.leave, None, error, prov.failed_operation)
                if prov.evidence is not None:
                    with prov.semantic.mute() if prov.semantic is not None else nullcontext():
                        prov.evidence.leave(name, None, error=error)
                raise
            else:
                if prov.semantic is not None:
                    prov.semantic.guard(prov.semantic.leave, output)
                if prov.evidence is not None:
                    with prov.semantic.mute() if prov.semantic is not None else nullcontext():
                        prov.evidence.leave(name, output)
                rec["out"] = _first_shape(output)
                # Logical footprint of one complete tensor output, not storage or
                # peak memory. Containers and repeated calls need richer evidence;
                # do not report the first tensor or last call as the whole output.
                if isinstance(output, torch.Tensor) and result.call_counts[name] == 1:
                    rec["output_tensor"] = {
                        "shape": list(output.shape),
                        "dtype": str(output.dtype).removeprefix("torch."),
                        "logicalBytes": output.numel() * output.element_size(),
                    }
                return output
            finally:
                prov.pop()
                if name == "":
                    prov.forward_end = len(prov.ops)

        mod._call_impl = call

        def remove():
            if mod.__dict__.get('_call_impl') is call:
                if own is missing:
                    del mod._call_impl
                else:
                    mod._call_impl = own
        handles.append(SimpleNamespace(remove=remove))

    for name, mod in model.named_modules():
        install(name, mod)
    if prov.semantic is not None:
        # Existing registered instances keep their exact wrappers, including custom
        # eager implementations. Observe dynamically constructed ordinary modules
        # through the class fallback without inventing a registered module path.
        original_class_call = nn.Module._call_impl
        def dynamic_call(mod, *args, **kwargs):
            if threading.get_ident() != prov.semantic.thread:
                prov.semantic.omit("unsupported_type")
                return original_class_call(mod, *args, **kwargs)
            prov.semantic.guard(prov.semantic.enter, mod, args, kwargs)
            try:
                output = original_class_call(mod, *args, **kwargs)
            except BaseException as error:
                prov.semantic.guard(prov.semantic.leave, None, error, prov.failed_operation)
                raise
            else:
                prov.semantic.guard(prov.semantic.leave, output)
                return output
        nn.Module._call_impl = dynamic_call
        def remove_class_call():
            if nn.Module._call_impl is dynamic_call:
                nn.Module._call_impl = original_class_call
        handles.append(SimpleNamespace(remove=remove_class_call))
    return handles


# ── runtime capture (task.md M8) ──────────────────────────────────────────────
# Statistics about the numbers flowing through the model, which the shapes alone
# cannot tell you: a layer whose activations have gone to NaN has exactly the same
# shape as one that is fine.
#
# Two properties this has to keep, and both are why the hooks below only *collect
# tensors* and compute nothing:
#
# * a forward hook fires INSIDE the ``_Provenance`` dispatch mode, so a ``mean()``
#   taken there is dispatched, recorded as an aten op owned by the module being
#   measured, and lands in the graph. Measuring would change the thing measured —
#   ``diagnostics.aten_ops`` and every dataflow edge derived from those ops;
# * the reductions are cheap but not free, and the trace is on the save path.
#
# So the hooks stash detached tensors and ``Capture.finish`` reduces them once the
# mode has exited. The number of tensors held is capped for the same reason the
# capture is opt-in at all: this holds real activations alive for the length of a
# forward, in the user's own process.

#: How many tensors one capture measures before it stops and says so. A tiny batch
#: through a large model is still thousands of modules, and a debugging aid that
#: can exhaust the user's memory is not one.
MAX_CAPTURED_TENSORS = 256


@dataclass
class CaptureSpec:
    """What the caller asked to be measured (``--capture``/``--loss``)."""

    activations: bool = False
    grads: bool = False
    #: A Python expression over ``out`` (the forward's return) and ``torch``,
    #: evaluated to the scalar the backward is run from. Required for ``grads``:
    #: there is no honest default — the loss is the user's, not ours.
    loss: Optional[str] = None
    max_tensors: int = MAX_CAPTURED_TENSORS
    #: The traced call runs its OWN backward (task.md M9's ``--kind train_step``:
    #: the user's ``train_step`` calls ``loss.backward()`` itself). The hooks are
    #: still wanted — they are what records the gradients — but running a second
    #: backward here would accumulate into ``.grad`` a second time and report a
    #: training step that never happened. ``loss`` is then not required, because
    #: the loss is the user's function's, not an expression we evaluate.
    external_backward: bool = False


def tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    """``{mean, std, absmax, nan, inf, dtype, device}`` for one tensor (§6.1).

    ``mean``/``std``/``absmax`` are computed over the FINITE elements only. The
    alternative — a plain ``.mean()`` — answers ``NaN`` the moment one element is
    NaN, which serialises to ``null`` and destroys the two useful numbers at the
    exact moment they are being looked at. The NaN is not hidden: ``nan`` and
    ``inf`` count the offending elements, and they are what turns the card amber.

    Integer and boolean tensors are reduced as float32; a token-id tensor has a
    meaningful mean and ``.mean()`` refuses to take it.
    """
    with torch.no_grad():
        flat = tensor.detach()
        nan = int(torch.isnan(flat).sum().item()) if flat.is_floating_point() else 0
        inf = int(torch.isinf(flat).sum().item()) if flat.is_floating_point() else 0
        values = flat.reshape(-1)
        if not values.is_floating_point():
            values = values.float()
        if nan or inf:
            values = values[torch.isfinite(values)]
        count = int(values.numel())
        mean = float(values.mean().item()) if count else 0.0
        std = float(values.std().item()) if count > 1 else 0.0
        absmax = float(values.abs().max().item()) if count else 0.0
    return {
        "mean": mean,
        "std": std,
        "absmax": absmax,
        "nan": nan,
        "inf": inf,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
    }


@dataclass
class Capture:
    """One capture's collected tensors, and the stats they reduce to.

    ``stats`` is keyed by the module's qualified path — the same key
    ``named_modules`` uses and the same one the graph carries as ``_module_path``
    — because the worker attaches the result to nodes by that path and nothing
    else in the envelope survives the identity rewrites in between.
    """

    spec: CaptureSpec
    stats: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    #: ``(kind, module path, tensor)``, reduced by ``finish``.
    _pending: List[Tuple[str, str, torch.Tensor]] = field(default_factory=list)
    truncated: bool = False

    def reset(self) -> None:
        """Forget an earlier attempt. ``_introspect_and_trace`` may call
        ``trace_model`` several times while it searches for an input that works;
        the run that produced the graph is the run whose numbers are reported."""
        self.stats = {}
        self._pending = []
        self.truncated = False

    def hold(self, kind: str, path: str, value: Any) -> None:
        tensors = _tensors(value)
        if not tensors:
            return
        if len(self._pending) >= self.spec.max_tensors:
            self.truncated = True
            return
        self._pending.append((kind, path, tensors[0].detach()))

    def watch_grad(self, path: str, value: Any) -> None:
        """Ask for the gradient of a module's output when the backward reaches it.

        A hook on the TENSOR, not ``register_full_backward_hook`` on the module,
        and the difference is measurable rather than stylistic: the module hook
        installs torch's ``BackwardHook``, which calls ``view_as`` on every input
        and output during the FORWARD. Those views dispatch, the tracer records
        them, and the graph grows nodes the model does not have — measured on the
        fixture net, ``--capture grads`` took it from 9 aten ops to 16 and added
        five phantom ``flatten`` nodes while losing the residual ``add``. A tensor
        hook adds nothing to the forward and delivers the same gradient.
        """
        tensors = _tensors(value)
        if not tensors or not tensors[0].requires_grad:
            return
        tensors[0].register_hook(lambda grad: self.hold("grad", path, grad))

    def finish(self, output: Any, failed: bool) -> None:
        """Run the backward (if asked), then reduce everything held.

        Called after the dispatch mode has exited, which is the whole point: the
        backward's own aten ops must not enter the traced graph, and a reduction
        taken inside the mode would be recorded as part of the model.

        ``external_backward`` skips the first half only: a ``--kind train_step``
        run already differentiated its own loss inside the user's function, and a
        second backward would double the gradients it is about to report.
        """
        if self.spec.grads and not failed and not self.spec.external_backward:
            self._backward(output)
        self.reduce()

    def reduce(self) -> None:
        """Turn every held tensor into its statistics and let it go.

        Split out of ``finish`` so a caller whose backward has already happened can
        ask for the numbers without asking for another one.
        """
        for kind, path, tensor in self._pending:
            try:
                self.stats.setdefault(path, {})[kind] = tensor_stats(tensor)
            except Exception as exc:  # a tensor no reduction applies to
                self.warnings.append(f"{path}: {kind} statistics failed ({exc})")
        self._pending = []
        if self.truncated:
            self.warnings.append(
                f"capture stopped after {self.spec.max_tensors} tensors; "
                "later modules carry no statistics"
            )

    def _backward(self, output: Any) -> None:
        """One backward pass from the user's own loss expression.

        The expression is required and evaluated here rather than guessed at
        (``output.sum()`` would silently answer for a model whose forward returns
        a tuple, logits and a cache, say). A failure is a warning and not an
        exception: the forward succeeded, the graph is real, and the caller is
        told which half is missing instead of losing both.
        """
        expression = (self.spec.loss or "").strip()
        if not expression:
            self.warnings.append("gradient capture needs --loss; none was given")
            return
        try:
            loss = eval(  # noqa: S307 - the worker runs the user's code by design
                compile(expression, "<loss>", "eval"), {"torch": torch}, {"out": output}
            )
        except Exception as exc:
            self.warnings.append(f"loss expression {expression!r} failed: {exc}")
            return
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            self.warnings.append(
                f"loss expression {expression!r} produced {type(loss).__name__}, "
                "not a single-element tensor"
            )
            return
        if not loss.requires_grad:
            self.warnings.append(
                f"loss expression {expression!r} has no grad_fn; nothing to differentiate"
            )
            return
        try:
            loss.backward()
        except Exception as exc:
            self.warnings.append(f"backward failed: {exc}")


#: The capture the next ``trace_model`` call should perform, or None.
#:
#: Process-global rather than an argument because the only caller that turns it on
#: — ``services/ingest/worker.py`` — reaches ``trace_model`` through
#: ``ingest_source``, and threading a parameter through would change a signature
#: three other importers pin. The worker is one trace per process (§5), so the
#: scope of this global is the scope of the run.
_ACTIVE: Optional[Capture] = None


@contextmanager
def capture_scope(spec: CaptureSpec) -> Iterator[Capture]:
    """Ask the next trace in this process to measure, and hand back what it found."""
    global _ACTIVE
    previous = _ACTIVE
    capture = Capture(spec)
    _ACTIVE = capture
    try:
        yield capture
    finally:
        _ACTIVE = previous


# ── the traced call (task.md M9) ──────────────────────────────────────────────
# A `--kind train_step` run traces the same model in the same way, and differs in
# exactly one place: what is CALLED. `model(*args)` becomes the user's own
# `train_step(model, batch, optimizer)`, which runs the forward, a backward and an
# optimiser step — so the graph is the model's, the gradients are a real training
# step's, and the loss is the user's own.
#
# Process-global for the same reason `_ACTIVE` is (see `capture_scope`): the only
# caller that sets it reaches `trace_model` through `service.ingest_source`, whose
# signature three other importers pin. Threading a parameter through would change
# all four to serve one.


@dataclass
class Step:
    """A call to run instead of the model's forward, and what it returned.

    ``call`` receives the model the tracer built, because the caller cannot: the
    model is constructed inside ``ingest_source``, several frames below whoever
    opened this scope.
    """

    call: Any
    returned: Any = None
    ran: bool = False

    def invoke(self, model: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        self.ran = True
        self.returned = self.call(model, args, kwargs)
        return self.returned


#: The call the next ``trace_model`` in this process should make, or None.
_STEP: Optional[Step] = None


@contextmanager
def step_scope(call: Any) -> Iterator[Step]:
    """Trace the next forward by calling ``call(model, args, kwargs)`` instead."""
    global _STEP
    previous = _STEP
    step = Step(call)
    _STEP = step
    try:
        yield step
    finally:
        _STEP = previous


@dataclass
class CallableTrace:
    """What one traced call did: its aten ops, and whether it finished."""

    ops: List[Dict[str, Any]] = field(default_factory=list)
    frames_captured: int = 0
    returned: Any = None
    error: Optional[str] = None
    error_frames: List[Tuple[str, int, str]] = field(default_factory=list)
    elapsed_ms: float = 0.0


def trace_callable(
    call: Any,
    *,
    frame_roots: Sequence[str] = (),
) -> "CallableTrace":
    """Record every aten op one plain Python call performed (task.md M9).

    The data-pipeline half of M9. There is no ``nn.Module`` here and therefore no
    module stack, no hooks and no hierarchy: a ``Dataset.__getitem__`` is a function,
    and the only provenance a function's tensor ops have is the line they ran on. So
    this is ``_Provenance`` with an empty stack — every op carries a ``frame`` when it
    ran under one of ``frame_roots``, and ``normalise.build_callable_graph`` turns
    those frames into nodes.

    Real tensors, never fake: a ``DataLoader`` reads files and decodes them, and a
    fake tensor cannot be written to a real buffer.
    """
    result = CallableTrace()
    prov = _Provenance(frame_roots)
    started = time.perf_counter()
    try:
        with torch.no_grad(), prov:
            result.returned = call()
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        result.error_frames = _frames_of(exc)
    finally:
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        result.ops = prov.ops
        result.frames_captured = prov.frames_captured
    return result


def _install_capture(model: nn.Module, capture: Capture) -> List[Any]:
    """One forward hook per named module: it holds the output and, for gradients,
    asks that output for its gradient when the backward comes back through it.

    The root module is skipped: it is the graph, not a card in it, and its output
    is already the output node's shape. The hook returns None — a forward hook
    that returns a value REPLACES the module's output (see ``_install_hooks``).
    """
    capture.reset()
    handles: List[Any] = []
    spec = capture.spec

    def make_hook(name: str):
        def _hook(mod, args, kwargs, output):
            if spec.activations:
                capture.hold("act", name, output)
            if spec.grads:
                capture.watch_grad(name, output)
            return None

        return _hook

    for name, mod in model.named_modules():
        if not name:
            continue
        handles.append(mod.register_forward_hook(make_hook(name), with_kwargs=True))
    return handles


class _AttentionFastPathOff:
    """PyTorch's fused attention fast path off while the traced call runs.

    ``nn.MultiheadAttention``, ``nn.TransformerEncoderLayer`` and the nested-tensor
    ``nn.TransformerEncoder`` switch to fused inference kernels
    (``_native_multi_head_attention``, ``_transformer_encoder_layer_fwd``) unless a
    module hook, a TorchFunctionMode or a tensor subclass is present. Whether a
    trace showed the sub-layers therefore depended on which instrumentation was
    active: the legacy real tier (capture hooks) drew ``linear1``/``norm1``/...,
    the same encoder on the fake tier drew one fused operation and lost their
    edges. With the flag off every tier records the decomposed operations, which
    is what the canonical recorder's function mode already forced.

    A plain ``__enter__``/``__exit__`` class rather than a wrapper function or a
    generator: it adds no frame to the traceback of an exception the model raises,
    so ``error_frames`` and the error site are exactly those of an untoggled call.
    Older torch has no switch; its fast path is left unchanged.
    """

    def __init__(self) -> None:
        mha = getattr(torch.backends, "mha", None)
        self._mha = mha if mha is not None and hasattr(mha, "set_fastpath_enabled") else None
        self._prior: bool | None = None

    def __enter__(self) -> None:
        if self._mha is not None:
            self._prior = self._mha.get_fastpath_enabled()
            self._mha.set_fastpath_enabled(False)

    def __exit__(self, *exc: Any) -> bool:
        if self._mha is not None and self._prior is not None:
            self._mha.set_fastpath_enabled(self._prior)
        return False


def fake_mode_active() -> bool:
    """Whether a ``FakeTensorMode`` is already on the dispatch stack.

    ``services/ingest/service.py`` builds and traces inside one, so entering a
    second here would put the model's tensors and the mode's tensors in different
    fake universes ("Please convert all Tensors to FakeTensors first"). The
    worker, which constructs for real, is not inside one — and that is exactly
    the difference between the two tiers.
    """
    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        return any(isinstance(m, FakeTensorMode) for m in _get_current_dispatch_mode_stack())
    except Exception:  # pragma: no cover - private API, availability differs
        return False


def trace_model(
    model: nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    tier: str = "fake",
    frame_roots: Sequence[str] = (),
    preserve_mode: bool = False,
    record_evidence: bool = False,
    grad_enabled: bool = False,
    semantic_context: Any = None,
) -> TraceResult:
    """Run one forward, capturing shapes, call counts and aten provenance.

    ``tier='fake'`` traces under FakeTensorMode (no storage). ``tier='real'`` runs on the
    tensors as given, which is the escalation for models whose control flow needs real
    values. A data-dependent failure is not fatal: whatever was captured before the
    exception is kept and the caller marks the remainder ``unreached``.

    Until M4 the two tiers were byte-identical branches and the fake-ness came
    entirely from the caller's surrounding ``FakeTensorMode``. They now are what
    they claim to be: ``fake`` enters the mode when nobody else has, ``real``
    deliberately does not — which is what lets ``if x.sum() > 0`` take a branch
    instead of raising ``DataDependentOutputException``. Callers already inside a
    fake mode (the library and repo importers) are unaffected: the mode is not
    entered twice and their trace is unchanged.

    ``frame_roots`` turns on per-op Python frame capture for ops running under
    those directories, which is what gives a functional node a ``file:line``.

    A ``capture_scope`` open in this process (task.md M8) additionally measures the
    activations, and with a loss expression the gradients, of every named module.
    It is real-tier only and adds one backward pass, both of which is why it is
    opt-in and off by default.
    """
    result = TraceResult()
    result.tier = tier
    prov = _Provenance(frame_roots)
    if semantic_context is not None and tier == "real":
        from ..semantic.recorder import SemanticRecorder
        prov.semantic = SemanticRecorder(model, semantic_context)
    if prov.semantic is None:
        result.input_ids = [prov.tensor_id(value) for value in (*args, *(kwargs or {}).values()) if isinstance(value, torch.Tensor)]
    if record_evidence and tier == "real":
        from .execution_evidence import ExecutionEvidence
        prov.evidence = ExecutionEvidence(model, args, kwargs or {})
    handles = _install_hooks(model, prov, result)
    # Real tier only (task.md M8): statistics about fake tensors are statistics
    # about nothing, and a backward under FakeTensorMode allocates no gradients.
    capture = _ACTIVE if tier == "real" else None
    capture_handles = _install_capture(model, capture) if capture is not None else []
    # What to call. `model(*args)` unless a step scope named something else — a
    # `--kind train_step` run traces the user's own train_step(model, batch, optimizer)
    # (task.md M9), which is the same forward with a backward and an optimiser step
    # around it. Real tier only: a step under FakeTensorMode allocates no gradients.
    step = _STEP if tier == "real" else None
    call = (
        (lambda: step.invoke(model, args, kwargs or {}))
        if step is not None
        else (lambda: model(*args, **(kwargs or {})))
    )
    # Inference mode for a plain forward. A module is born in train mode, and the
    # graph this produces is the model's inference structure, so tracing in train
    # mode is wrong twice over: BatchNorm1d raises outright on a batch of one
    # ("Expected more than 1 value per channel when training") for a model that is
    # perfectly correct, and Dropout puts noise in a trace that is meant to be
    # deterministic. A `--kind train_step` run is the deliberate exception — its
    # whole point is the training path — so it keeps the mode the user's own code set.
    was_training = model.training
    if step is None and was_training and not preserve_mode:
        model.eval()
    started = time.perf_counter()
    output: Any = None
    forward_error: BaseException | None = None
    try:
        if tier == "fake" and not fake_mode_active():
            with _AttentionFastPathOff(), torch.no_grad(), FakeTensorMode(allow_non_fake_inputs=True), prov:
                output = call()
        elif step is not None or (capture is not None and capture.spec.grads):
            # The one path that does not run under no_grad: a backward needs the
            # graph the forward builds, and no_grad refuses to build it. A step
            # scope is here for the same reason — its whole point is the backward.
            with _AttentionFastPathOff(), prov.semantic.functions() if prov.semantic is not None else nullcontext(), prov:
                output = call()
        else:
            with _AttentionFastPathOff(), torch.set_grad_enabled(grad_enabled), prov.semantic.functions() if prov.semantic is not None else nullcontext(), prov:
                output = call()
    except DATA_DEPENDENT as exc:  # partial trace is still useful
        forward_error = exc
        result.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        result.error_frames = _frames_of(exc)
    except Exception as exc:
        forward_error = exc
        result.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        result.error_frames = _frames_of(exc)
    finally:
        if step is None and was_training and not preserve_mode:
            model.train()
        for handle in handles:
            handle.remove()
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        if step is None:
            result.ops = prov.ops
            result.frames_captured = prov.frames_captured
        else:
            # The graph a training step produces is the MODEL's graph. Everything the
            # step did around the forward — `zero_grad`, the backward's gradient
            # accumulation, the optimiser's in-place parameter updates — is real work
            # and none of it is a layer, so it is measured (diagnostics.loss and
            # `_grad_stats`) rather than drawn. Without this the fixture net grew
            # eight `add` cards that exist in no version of that model: gradient
            # accumulation is a chain of two-producer adds, and that is exactly the
            # shape `resolve_dataflow` materialises a residual from.
            start = prov.forward_start or 0
            end = prov.forward_end if prov.forward_end is not None else len(prov.ops)
            result.ops = prov.ops[start:end]
            result.frames_captured = sum(1 for op in result.ops if op.get("frame"))
        # A trace cut short by a data-dependent failure leaves frames open, so drain
        # them: an unclosed frame still holds real reuse evidence.
        while len(prov._frames) > 1:
            prov.pop()
        result.reuse_scopes = list(prov.reuse)
        if prov.evidence is not None:
            result.evidence = prov.evidence.snapshot(forward_error)
            if result.error is None:
                from .execution_values import capture_output
                result.evidence['outputs'] = capture_output(output)
        if forward_error is not None:
            from ..semantic.diagnostics import forward_diagnostic
            result.diagnostic = forward_diagnostic(forward_error, prov.failed_operation)
        prov.failed_operation = None
        if prov.semantic is not None:
            result.semantic_capture = prov.semantic.guard(prov.semantic.snapshot)
            if result.semantic_capture is None:
                result.semantic_error = "recorder_error" if "recorder_error" in prov.semantic.omissions else "byte_limit"
            elif result.semantic_capture["status"] == "failed" and result.error is None:
                result.semantic_error = "recorder_error"
                result.error = "CaptureRecorderError: the forward completed but its invocation could not be recorded."
    if capture is not None:
        # Outside the `finally`, and outside the dispatch mode it restored: the
        # backward's aten ops would otherwise be recorded as part of the model's
        # forward and rewrite its edges. The forward hooks are still registered
        # while this runs, which is harmless — they only fire on a forward — and
        # removing a full backward hook before the backward would silence it.
        try:
            capture.finish(output, failed=result.error is not None)
        finally:
            for handle in capture_handles:
                handle.remove()
    return result


def _frames_of(exc: BaseException) -> List[Tuple[str, int, str]]:
    """``(file, line, function)`` for every frame of a failed forward, innermost last."""
    return [
        (frame.filename, int(frame.lineno or 0), frame.name)
        for frame in traceback.extract_tb(exc.__traceback__)
    ]


def dataflow_edges(
    ops: List[Dict[str, Any]], rendered: Set[str], parent: Dict[str, str]
) -> Set[Tuple[str, str]]:
    """The original fold, unchanged: edges only, nothing materialised.

    Kept as the pinned contract. With no plan the two new branches in
    ``resolve_dataflow`` are unreachable, so the transparency rules below are still
    exercised by the tests that guard them.
    """
    return set(resolve_dataflow(ops, rendered, parent))


def resolve_dataflow(
    ops: List[Dict[str, Any]],
    rendered: Set[str],
    parent: Dict[str, str],
    plan: Any = None,
    seeds: Optional[Dict[Any, str]] = None,
) -> List[Tuple[str, str]]:
    """Fold tensor-level provenance up to the deepest RENDERED ancestor.

    Two rules that are easy to get wrong and produce silently wrong graphs:

    * A module that has rendered CHILDREN is a *frame*, not a node. The functional ops in
      its own forward (the ``wte + wpe`` add, a residual, a ``cat``) must be TRANSPARENT
      and propagate their inputs' producers instead of claiming authorship. Otherwise
      every sibling chain snaps at the parent — gpt2 drops from a full chain to 2 edges.
    * ``producer[tensor]`` must be a SET. A residual add or ``torch.cat`` has several
      contributing producers and collapsing to one drops real edges (gpt2 would lose
      ``wpe -> drop`` entirely).

    With a ``plan`` (see materialise.py) two things change, and ONLY for ops the rules
    above would otherwise have discarded:

    * a whitelisted functional op with two or more distinct producers becomes its own
      merge node instead of unioning them into its consumer. Unioning is what gave a
      DenseNet's norm 17 inbound edges and a MobileNet's conv 2.
    * a rendered leaf entered more than once inside one invocation of its parent gets a
      distinct node per call site, which is what breaks the ResNet cycle.

    Nothing is materialised for an op whose whole subtree is unrendered (``fold`` returns
    None and the owner is not the model root). Those ops belong to a branch the depth limit
    hid — a MobileNetV3 squeeze-excitation, whose pool and multiply are visible while the
    two convolutions between them are not — so giving them top-level nodes spliced a
    half-branch into the main chain and broke the channel counts. They stay transparent.

    ``seeds`` names the producer of a tensor nobody in the trace produced — the
    model's own arguments. Without it a residual against the INPUT
    (``return x + self.l2(h)``) has one known producer instead of two, the merge
    rule declines, and the add silently disappears from the graph: the fixture
    net renders as a straight chain with no residual at all. It is opt-in because
    it changes the edge set, and the library and repository importers are pinned
    on the edge set they have.

    Everything else stays on the byte-identical transparent path.
    """
    has_child = {
        p for p in rendered if any(c != p and parent.get(c) == p for c in rendered)
    }

    def fold(name: str) -> Optional[str]:
        if name in rendered:
            return name
        parts = name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in rendered:
                return candidate
        return None

    producer: Dict[Any, Set[str]] = {tid: {name} for tid, name in (seeds or {}).items()}
    pairs: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    def emit(src: str, dst: str) -> None:
        if src != dst and (src, dst) not in seen:
            seen.add((src, dst))
            pairs.append((src, dst))

    for op in ops:
        raw_owner = op["owner"]
        owner = fold(raw_owner)
        sources: Set[str] = set()
        for tid in op["in_ids"]:
            sources |= producer.get(tid, set())
        if owner is not None and owner not in has_child:
            if (
                plan is not None
                and len(sources) >= 2
                and plan.wants_merge(op)
                and plan.is_composite(owner)
                # If the composite is its OWN producer the merge is internal — a
                # BasicBlock at depth 1 adds its own output to the identity, so
                # materialising would emit layer1.0 -> add -> layer1.0 and make the
                # graph cyclic. Measured: doing so turned four clean depth-1 imports
                # cyclic. Let the card own that add, as it did before.
                and owner not in sources
            ):
                # The merge runs inside a COMPOSITE whose children are not rendered — a
                # DenseNet _DenseLayer at depth 1 concatenating every earlier feature map.
                # Letting the composite claim the op gave it up to 17 inbound edges; the
                # concat is real structure, so give it a node and feed the composite from
                # it. A merge inside a genuine LEAF is left alone: a ReLU contains no add.
                merged = plan.merge_node(op, owner, sorted(sources))
                for src in sources:
                    emit(src, merged)
                emit(merged, owner)
                for tid in op["out_ids"]:
                    producer[tid] = {owner}
                continue
            node = owner
            if plan is not None and owner == raw_owner:
                # Only the owner itself can be split; a folded ancestor's call index
                # says nothing about this op.
                node = plan.call_site(owner, op.get("owner_call", 1), op)
            for src in sources:
                emit(src, node)  # a loop-called module must not self-edge
            for tid in op["out_ids"]:
                producer[tid] = {node}
        elif plan is not None and len(sources) >= 2 and plan.wants_merge(op):
            merged = plan.merge_node(op, owner, sorted(sources))
            for src in sources:
                emit(src, merged)
            for tid in op["out_ids"]:
                producer[tid] = {merged}
        elif (
            plan is not None
            and len(sources) >= 1
            and plan.wants_pool(op)
            and (owner is not None or raw_owner == "")
        ):
            pooled = plan.pool_node(op, owner, sorted(sources))
            for src in sources:
                emit(src, pooled)
            for tid in op["out_ids"]:
                producer[tid] = {pooled}
        elif plan is not None and len(sources) == 1 and plan.wants_flatten(op):
            flat = plan.flatten_node(op, owner, sorted(sources))
            for src in sources:
                emit(src, flat)
            for tid in op["out_ids"]:
                producer[tid] = {flat}
        elif sources:
            for tid in op["out_ids"]:
                producer[tid] = producer.get(tid, set()) | sources
    return pairs

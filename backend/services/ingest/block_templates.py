"""Within-run block templates ("keep X") for the canonical fake-tier capture.

Off unless ``TENSORVIZ_BLOCK_TEMPLATES`` is ``1`` or ``shadow`` when the worker
starts. Only a canonical ``SemanticRecorder`` capture on the fake tier builds a
cache (``trace_model``); the real tier, the legacy fake trace and every other
path are untouched.

The first call of a candidate module with a given key runs live. While it runs,
the recorder's own observations are journaled: every value fact it appends,
every source-binding lookup (inside or outside the call), every nested hooked
call. A later call with an equal key and an equal *slot state* (which of its
argument/state tensors the recorder already knows) is stamped: the
invocations, operations and value facts the live call would append are
written straight into the recorder with ids remapped from the template, the
recorder's identity tables are advanced exactly as ``observe`` would have, and
fake outputs with the recorded metadata are returned. Nothing is replayed
through the recorder's event methods.

Exactness is protected by: an ineligibility deny-list (writes outside the
call, RNG, value reads, attribute/python-state writes, hooks, modules built in
forward, exceptions, any recorder omission), a byte model that must reproduce
the recorder's own accounting on the recorded call, pre-checks before any
mutation (limit headroom, parameter-count budget, module identities, frame
depth), and ``shadow`` mode, which stamps and records live and compares both.
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import types
import weakref
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..semantic.identity import weak_get, weak_set

#: Read once, when the worker imports the trace module.
MODE = os.environ.get("TENSORVIZ_BLOCK_TEMPLATES", "")
ENABLED = MODE in ("1", "shadow")
SHADOW = MODE == "shadow"

_PRIM = (bool, int, float, complex, str, bytes, type(None), torch.dtype, torch.device,
         torch.layout, torch.memory_format)
_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
               "_forward_hooks_with_kwargs", "_forward_pre_hooks_with_kwargs", "_forward_hooks_always_called")
_SKIP_ATTRS = frozenset({
    "_parameters", "_buffers", "_modules", "_call_impl", "training",
    "_non_persistent_buffers_set", "_state_dict_hooks", "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks", "_load_state_dict_post_hooks",
    "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
    "_forward_hooks_with_kwargs", "_forward_pre_hooks_with_kwargs", "_forward_hooks_always_called",
    "_is_full_backward_hook",
})
_VALUE_READS = ("_local_scalar_dense", "is_nonzero", "aten.item", "aten.equal", "allclose")
_INDEX_TENSOR = torch.ops.aten.index.Tensor
_GC_GEN2 = 1_000_000


class Ineligible(Exception):
    pass


class _StampError(Exception):
    pass


# ── key helpers (ported from the legacy-path spike) ───────────────────────────

def _global_hooks_present() -> bool:
    m = nn.modules.module
    for attr in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
                 "_global_backward_pre_hooks", "_global_forward_hooks_always_called"):
        if getattr(m, attr, None):
            return True
    return False


def _tmeta(t: torch.Tensor) -> tuple:
    if t.layout is not torch.strided:
        raise Ineligible("layout")
    return (type(t), tuple(t.shape), t.stride(), t.storage_offset(), t.dtype, t.device, t.requires_grad,
            isinstance(t, nn.Parameter))


def _flatten(value: Any, out: List[torch.Tensor]) -> Any:
    """Structure spec of an input/output tree; tensors appended to ``out``."""
    if isinstance(value, torch.Tensor):
        out.append(value)
        return ("T",)
    tv = type(value)
    if tv is tuple or tv is list:
        return (tv, tuple(_flatten(v, out) for v in value))
    if tv is torch.Size:
        return (tv, tuple(value))
    if tv is dict:
        items = []
        for k, v in value.items():
            if type(k) not in _PRIM:
                raise Ineligible("opaque_key")
            items.append((k, _flatten(v, out)))
        return (dict, tuple(items))
    if isinstance(value, _PRIM) and tv in _PRIM:
        return (tv, value)
    raise Ineligible("opaque_value")


def _bad_tensor_in_container():
    raise Ineligible("tensor_in_container_attr")


def _enc_attr(v: Any, path: str, attr_tensors: List[Tuple[str, torch.Tensor]]) -> Any:
    tv = type(v)
    if isinstance(v, _PRIM):
        return (tv, v)
    if isinstance(v, torch.Tensor):
        attr_tensors.append((path, v))
        return ("tensor", _tmeta(v))
    if tv is tuple or tv is list or tv is torch.Size:
        return (tv, tuple(_enc_attr(x, path + "[]", []) if not isinstance(x, torch.Tensor)
                          else _bad_tensor_in_container() for x in v))
    if tv is dict:
        return (tv, tuple((k if type(k) in _PRIM else ("id", id(k)),
                           _enc_attr(x, path + "{}", []) if not isinstance(x, torch.Tensor)
                           else _bad_tensor_in_container()) for k, x in v.items()))
    if tv is set or tv is frozenset:
        if all(type(x) in _PRIM for x in v):
            return (tv, frozenset(v))
        return ("id", id(v))
    return _shallow(v)


def _torch_rng_bytes() -> bytes:
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes():
        return torch.random.get_rng_state().numpy().tobytes()


def _rng_unused(func: Any, args: tuple, kwargs: dict) -> bool:
    """A seeded-capable op whose probability argument is exactly 0 (or train=False).

    SDPA carries ``nondeterministic_seeded`` even at ``dropout_p=0``."""
    try:
        schema = func._schema.arguments
    except Exception:
        return False
    for i, a in enumerate(schema):
        if a.name in ("dropout_p", "p", "train"):
            v = args[i] if i < len(args) else kwargs.get(a.name, a.default_value if a.has_default_value() else None)
            if a.name == "train":
                return v is False
            return type(v) in (int, float) and v == 0
    return False


def _integer_indices(args: tuple) -> bool:
    """aten.index.Tensor is tagged dynamic_output_shape for BOOL masks only."""
    if len(args) < 2 or not isinstance(args[1], (list, tuple)):
        return False
    seen = False
    for t in args[1]:
        if t is None:
            continue
        if not isinstance(t, torch.Tensor) or t.dtype not in (torch.int64, torch.int32):
            return False
        seen = True
    return seen


_TORCH_DIR = os.path.dirname(torch.__file__)
_STDLIB = tuple({os.path.realpath(p) for p in (sys.base_prefix, sys.prefix)})
_SITE = ("site-packages", "dist-packages")
_MAX_REFS = 512


def _user_code(code: types.CodeType) -> bool:
    f = code.co_filename
    if f.startswith("<") or f.startswith(_TORCH_DIR):
        return False
    if any(s in f for s in _SITE):
        return "/torch/" not in f and "/numpy/" not in f
    return not os.path.realpath(f).startswith(_STDLIB)


def _codes(code: types.CodeType):
    yield code
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            yield from _codes(c)


def _shallow(v: Any, depth: int = 2) -> Any:
    """Value encoding of python state that can steer a forward."""
    tv = type(v)
    if isinstance(v, _PRIM):
        return (tv, v)
    if isinstance(v, (torch.Tensor, nn.Module, types.ModuleType, types.FunctionType,
                      types.BuiltinFunctionType, types.MethodType, type)) or depth <= 0:
        return ("id", id(v))
    if tv in (list, tuple, set, frozenset):
        if len(v) > 64:
            return (tv, "len", len(v), id(v))
        items = sorted(v, key=repr) if tv in (set, frozenset) else v
        return (tv, tuple(_shallow(x, depth - 1) for x in items))
    if tv is dict:
        if len(v) > 64:
            return (tv, "len", len(v), id(v))
        return (tv, tuple((k if type(k) in _PRIM else ("id", id(k)), _shallow(x, depth - 1)) for k, x in v.items()))
    d = getattr(v, "__dict__", None)
    if type(d) is dict:
        return (tv, id(v), _shallow(d, depth))
    return ("id", id(v))


class _Refs:
    __slots__ = ("lookups", "cells", "classes", "rng_np", "rng_py")


def _touches(v: Any, lib: str) -> bool:
    for o in (v, getattr(v, "__self__", None)):
        if o is None:
            continue
        mn = o.__name__ if isinstance(o, types.ModuleType) else getattr(o, "__module__", None) or type(o).__module__
        if isinstance(mn, str) and (mn == lib or mn.startswith(lib + ".") or (lib == "random" and mn == "_random")):
            return True
    return False


def _class_refs(cls: type, cache: Dict[type, Any]) -> Any:
    r = cache.get(cls)
    if r is not None:
        return r
    lookups: Dict[Tuple[int, str], Tuple[dict, str]] = {}
    cells: Dict[int, Any] = {}
    classes: Dict[int, type] = {}
    seen_fn: set = set()
    stack: List[Any] = []
    for k in cls.__mro__:
        if k in (nn.Module, object):
            continue
        classes[id(k)] = k
        for v in k.__dict__.values():
            f = v.__func__ if isinstance(v, (staticmethod, classmethod)) else v
            if isinstance(f, property):
                f = f.fget
            if isinstance(f, types.FunctionType):
                stack.append(f)
    while stack:
        f = stack.pop()
        if id(f) in seen_fn or not _user_code(f.__code__):
            continue
        seen_fn.add(id(f))
        if len(lookups) + len(cells) > _MAX_REFS:
            raise Ineligible("py_refs_cap")
        g = f.__globals__
        for cell in f.__closure__ or ():
            cells[id(cell)] = cell
            try:
                cv = cell.cell_contents
            except ValueError:
                continue
            if isinstance(cv, types.FunctionType):
                stack.append(cv)
        for code in _codes(f.__code__):
            for n in code.co_names:
                if n in g:
                    lookups[(id(g), n)] = (g, n)
                    gv = g[n]
                    if isinstance(gv, types.FunctionType):
                        stack.append(gv)
                    elif isinstance(gv, type) and not issubclass(gv, nn.Module) and gv.__module__ == g.get("__name__"):
                        classes[id(gv)] = gv
                        for cv in gv.__dict__.values():
                            if isinstance(cv, types.FunctionType):
                                stack.append(cv)
    r = _Refs()
    vals = [g.get(n) for g, n in lookups.values()]
    for c in cells.values():
        try:
            vals.append(c.cell_contents)
        except ValueError:
            pass
    r.rng_np = any(_touches(v, "numpy") for v in vals)
    r.rng_py = any(_touches(v, "random") for v in vals)
    r.lookups = tuple(lookups.values())
    r.cells = tuple(cells.values())
    r.classes = tuple(classes.values())
    cache[cls] = r
    return r


_ABSENT = ("absent",)


def _class_state(k: type) -> tuple:
    out = []
    for n, v in k.__dict__.items():
        if n.startswith("__") or isinstance(v, (types.FunctionType, staticmethod, classmethod, property)):
            continue
        out.append((n, _shallow(v, 1)))
    return tuple(out)


def _refs_state(refs: Any) -> tuple:
    vals = []
    for g, n in refs.lookups:
        vals.append(_shallow(g.get(n, _ABSENT)))
    for c in refs.cells:
        try:
            vals.append(_shallow(c.cell_contents))
        except ValueError:
            vals.append(None)
    for k in refs.classes:
        vals.append(_class_state(k))
    return tuple(vals)


def _version(t: torch.Tensor):
    try:
        return t._version
    except RuntimeError:
        return None


# ── JSON size model ───────────────────────────────────────────────────────────

def _size(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def _jlen(v) -> int:
    """Length of one scalar token in the recorder's compact JSON."""
    if v is None:
        return 4
    if type(v) is int:
        return len(str(v))
    if type(v) is list:
        return _size(v)  # a sourceBindingIds list
    return len(v) + 2  # ids are ASCII without escapes


class _V:
    """A variable token inside a record skeleton."""
    __slots__ = ("sym",)

    def __init__(self, sym):
        self.sym = sym


def _has_var(x) -> bool:
    if type(x) is _V:
        return True
    if type(x) is dict:
        return any(_has_var(v) for v in x.values())
    if type(x) is list:
        return any(_has_var(v) for v in x)
    return False


def _freeze(x):
    """Mark constant subtrees so instantiation can share them."""
    if type(x) is dict:
        if not _has_var(x):
            return ("C", x)
        return ("D", tuple((k, _freeze(v)) for k, v in x.items()))
    if type(x) is list:
        if not _has_var(x):
            return ("C", x)
        return ("L", tuple(_freeze(v) for v in x))
    if type(x) is _V:
        return ("V", x.sym)
    return ("C", x)


def _inst(node, resolve, acc):
    tag, body = node
    if tag == "C":
        return body
    if tag == "V":
        v = resolve(body)
        acc[0] += _jlen(v)
        return v
    if tag == "D":
        return {k: _inst(v, resolve, acc) for k, v in body}
    return [_inst(v, resolve, acc) for v in body]


def _fixed(node, resolve) -> int:
    acc = [0]
    value = _inst(node, resolve, acc)
    return _size(value) - acc[0]


# ── state containers ──────────────────────────────────────────────────────────

class _Ctx:
    __slots__ = ("flat", "arg_index", "state_obj", "state_path", "rel_mod", "mod_rel", "py", "args_spec",
                 "rng_flags", "slots", "slot_storage")

    def __init__(self):
        self.flat: List[torch.Tensor] = []
        self.arg_index: Dict[int, int] = {}
        self.state_obj: Dict[str, torch.Tensor] = {}
        self.state_path: Dict[int, str] = {}
        self.rel_mod: Dict[str, nn.Module] = {}
        self.mod_rel: Dict[int, str] = {}
        self.py: Any = None
        self.args_spec: Any = None
        self.slots: List[torch.Tensor] = []
        self.slot_storage: List[int] = []


class Recording:
    __slots__ = ("key", "mod", "name", "ctx", "frame", "start", "failed", "snap", "rng", "ext_storages",
                 "args", "root", "nv0", "no0", "ni0", "seq0", "bytes0", "bb0", "nb0", "pc0", "obj0", "sto0",
                 "nmod0", "ndef0", "ndiag0", "slot_pre", "epochs_pre", "cov0")


class Template:
    __slots__ = ("facts", "records", "events", "out_spec", "out_slots", "n_facts", "n_ops", "n_inv", "ticks",
                 "pc_delta", "obj_delta", "sto_delta", "bytes_tpl", "n_tokens", "u_depth", "epochs",
                 "ops_seen", "inv_seen", "rels", "has_u", "cls")


class _Fact:
    __slots__ = ("ext", "oid", "sid", "oid_mint", "sid_mint", "static", "wepoch", "sepoch", "producer",
                 "fixed", "prod_fixed")


# ── the cache ─────────────────────────────────────────────────────────────────

class TemplateCache:
    def __init__(self, prov: Any, result: Any, model: nn.Module) -> None:
        self.prov = prov
        self.result = result
        self.rec = prov.semantic
        self.values = self.rec.values
        self.shadow = SHADOW
        self.names: Dict[int, str] = {id(m): n for n, m in model.named_modules()}
        self.templates: Dict[Any, Template] = {}
        self.bad_keys: Dict[Any, str] = {}
        self.active: List[Recording] = []
        # id(call wrapper frame) -> its recording. Ids only: a held frame would keep
        # the call's locals (and so interior tensors) alive, which the recorder sees.
        self.boundaries: Dict[int, Recording] = {}
        self.log: List[tuple] = []
        self.paused = False
        self.suppress = False
        self.stats: Counter = Counter()
        self.mismatches: List[Dict[str, str]] = []
        self._ops: Dict[Any, tuple] = {}
        self._refs: Dict[type, Any] = {}
        intern: Dict[Any, int] = {}
        memo: Dict[int, int] = {}

        def sig(m: nn.Module) -> int:
            s = memo.get(id(m))
            if s is not None:
                return s
            key = (type(m),
                   tuple((n, None if c is None else sig(c)) for n, c in m._modules.items()),
                   tuple((n, None if p is None else (tuple(p.shape), p.dtype)) for n, p in m._parameters.items()),
                   tuple((n, None if b is None else (tuple(b.shape), b.dtype)) for n, b in m._buffers.items()))
            s = intern.setdefault(key, len(intern))
            memo[id(m)] = s
            return s

        groups: Counter = Counter()
        seen = set()
        for _, m in model.named_modules():
            if id(m) in seen:
                continue
            seen.add(id(m))
            groups[sig(m)] += 1
        self.static_repeat = {mid: groups[s] >= 2 for mid, s in memo.items()}
        self._install()

    # ── recorder instrumentation (instance attributes, removed by close) ─────
    def _install(self) -> None:
        rec, values = self.rec, self.values
        cache = self
        self._orig_observe = values.observe
        self._orig_omit = rec.omit

        def observe(tensor):
            if not cache.active:
                return cache._orig_observe(tensor)
            n0, b0 = len(values.values), rec.bytes
            fact = cache._orig_observe(tensor)
            if len(values.values) != n0:
                oid = fact["objectId"]
                cache.log.append(("V", fact, id(tensor), values.write_epochs.get(oid, 0),
                                  values.storage_epochs.get(fact["storageId"], 0), rec.bytes - b0))
            return fact

        def omit(reason):
            if cache.active:
                cache.fail_all("omission:" + str(reason))
            return cache._orig_omit(reason)

        values.observe = observe
        rec.omit = omit
        rec.encoder.omit = omit
        values.omit = omit
        self._sources = rec.sources
        if self._sources is not None:
            from ..semantic import source_bindings
            sources = self._sources
            max_frames = source_bindings.MAX_FRAMES
            internal = sources.internal
            torch_dir = sources.torch_directory
            expression = sources.expression

            def call_binding():
                # Exactly CapturedSourceEvidence.call_binding, plus which open
                # recording boundaries the walk passed and at what distance.
                frame = sys._getframe(1)
                bounds = cache.boundaries
                passed = None
                try:
                    for i in range(max_frames):
                        if frame is None:
                            break
                        if bounds:
                            # A boundary frame is executing for the whole recording,
                            # so its id cannot be reused while it is in this table.
                            b = bounds.get(id(frame))
                            if b is not None:
                                (passed := passed or []).append((b, i))
                        filename = frame.f_code.co_filename
                        if filename not in internal and not filename.startswith(torch_dir):
                            found = expression(frame.f_code, frame.f_lasti)
                            if cache.active:
                                cache.log.append(("B", found, passed, False))
                            return found
                        frame = frame.f_back
                finally:
                    del frame
                if cache.active:
                    cache.log.append(("B", None, passed, True))
                return None

            sources.call_binding = call_binding

    def close(self) -> None:
        values, rec = self.values, self.rec
        for obj, name in ((values, "observe"), (rec, "omit"), (rec.encoder, "omit"), (values, "omit")):
            obj.__dict__.pop(name, None)
        rec.encoder.omit = rec.omit  # bound method of the class again
        values.omit = rec.omit
        if self._sources is not None:
            self._sources.__dict__.pop("call_binding", None)
        self.active.clear()
        self.boundaries.clear()
        self.log.clear()

    def diagnostics(self) -> Dict[str, Any]:
        s = self.stats
        reasons = {k.split(":", 1)[1]: v for k, v in sorted(s.items()) if k.startswith("inelig:")}
        prechecks = {k.split(":", 1)[1]: v for k, v in sorted(s.items()) if k.startswith("precheck:")}
        nokeys = {k.split(":", 1)[1]: v for k, v in sorted(s.items()) if k.startswith("nokey:")}
        out = dict(mode="shadow" if self.shadow else "on", hits=s["hits"], misses=s["misses"],
                   templates=s["templates"], stampedOperations=s["stamped_ops"],
                   ineligible=reasons, prechecks=prechecks, noKey=nokeys,
                   skippedBadKey=s["skip_bad_key"], liveWhileRecording=s["live_while_recording"])
        errors = {k.split(":", 1)[1]: v for k, v in sorted(s.items()) if k.startswith("internal_error:")}
        if errors:
            out["internalErrors"] = errors
        if self.shadow:
            out.update(shadowCompared=s["shadow_compared"], shadowMismatches=s["shadow_mismatch"],
                       shadowMismatchSamples=list(self.mismatches))
        return out

    # ── decision ──────────────────────────────────────────────────────────────
    def candidate(self, name: str, mod: nn.Module) -> bool:
        if name == "" or not mod._modules:
            return False
        if self.static_repeat.get(id(mod)):
            return True
        return self.result.call_counts.get(name, 0) >= 2

    def plan(self, name: str, mod: nn.Module, args: tuple, kwargs: dict, frame: Any):
        if not self.candidate(name, mod):
            return None
        rec = self.rec
        if rec.stopped or rec.finished or not rec.stack or rec.stack[-1] is None or rec.pending is not None:
            self.stats["precheck:recorder_state"] += 1
            return None
        prior_s, prior_b = self.suppress, rec.busy
        self.suppress, rec.busy = True, True
        try:
            key, ctx = self.key(name, mod, args, kwargs)
        except Ineligible as why:
            self.stats["nokey:" + str(why)] += 1
            return None
        finally:
            self.suppress, rec.busy = prior_s, prior_b
        tpl = self.templates.get(key)
        if tpl is not None:
            if self.active:
                self.stats["live_while_recording"] += 1
                return None
            return ("hit", tpl, ctx)
        if key in self.bad_keys:
            self.stats["skip_bad_key"] += 1
            return None
        r = self.begin(key, mod, name, ctx, frame)
        r.args = (args, kwargs)
        return ("rec", r)

    def slot_state(self, t: torch.Tensor) -> tuple:
        values = self.values
        oid = weak_get(values.objects, t)
        st = t.untyped_storage()
        sid = weak_get(values.storages, st)
        version = _version(t)
        snap = None
        if oid is not None:
            key = values.snapshot_key(oid, version)
            fact = values.snapshots.get(key)
            if fact is not None:
                meta = (list(t.shape), list(t.stride()), t.storage_offset(), t.requires_grad, sid)
                ok = values.metadata[key] == meta and values.snapshot_epochs[key] == values.storage_epochs.get(sid, 0)
                snap = fact["id"] if ok else False
        return oid, sid, snap, version

    def key(self, name: str, mod: nn.Module, args: tuple, kwargs: dict):
        if _global_hooks_present():
            raise Ineligible("global_hooks")
        ctx = _Ctx()
        flat = ctx.flat
        spec = (_flatten(args, flat), _flatten(kwargs, flat))
        ctx.arg_index = {id(t): i for i, t in reversed(list(enumerate(flat)))}
        mods = []
        seen_state: Dict[int, int] = {}
        names = self.names
        for rel, m in mod.named_modules(remove_duplicate=False):
            if rel in ctx.rel_mod:
                continue
            if rel:
                installed = names.get(id(m))
                if installed != f"{name}.{rel}":
                    raise Ineligible("foreign_module")
            ctx.rel_mod[rel] = m
            ctx.mod_rel.setdefault(id(m), rel)
            d = m.__dict__
            for h in _HOOK_DICTS:
                if d.get(h):
                    raise Ineligible("user_hooks")
            attrs = []
            attr_tensors: List[Tuple[str, torch.Tensor]] = []
            for k, v in d.items():
                if k in _SKIP_ATTRS:
                    continue
                attrs.append((k, _enc_attr(v, k, attr_tensors)))
            pre = rel + "." if rel else ""
            ps = []
            for pn, p in m._parameters.items():
                if p is None:
                    ps.append((pn, None))
                    continue
                tie = seen_state.setdefault(id(p), len(seen_state))
                path = pre + pn
                ctx.state_obj.setdefault(path, p)
                ctx.state_path.setdefault(id(p), path)
                ps.append((pn, _tmeta(p), tie))
            bs = []
            for bn, b in m._buffers.items():
                if b is None:
                    bs.append((bn, None))
                    continue
                tie = seen_state.setdefault(id(b), len(seen_state))
                path = pre + bn
                ctx.state_obj.setdefault(path, b)
                ctx.state_path.setdefault(id(b), path)
                bs.append((bn, _tmeta(b), tie))
            for an, t in attr_tensors:
                tie = seen_state.setdefault(id(t), len(seen_state))
                path = pre + an
                ctx.state_obj.setdefault(path, t)
                ctx.state_path.setdefault(id(t), path)
                bs.append(("@" + an, tie))
            tm = type(m)
            mods.append((rel, tm, getattr(tm, "forward", None), m.training, tuple(attrs), tuple(ps), tuple(bs)))
        for t in flat:
            if id(t) in ctx.state_path:
                raise Ineligible("state_passed_as_arg")
        # Slots: every argument tensor, then every state tensor. Their metadata,
        # identity/storage aliasing pattern and what the recorder already knows
        # about them are all part of the lookup key.
        slots = list(flat) + list(ctx.state_obj.values())
        ctx.slots = slots
        ident: Dict[int, int] = {}
        stor: Dict[int, int] = {}
        sig = []
        for i, t in enumerate(slots):
            j = ident.setdefault(id(t), i)
            try:
                sc = t.untyped_storage()._cdata
            except Exception:
                raise Ineligible("storage")
            k = stor.setdefault(sc, i)
            ctx.slot_storage.append(sc)
            oid, sid, snap, version = self.slot_state(t)
            if snap is False:
                raise Ineligible("stale_snapshot")
            sig.append((_tmeta(t), j, k, oid is not None, sid is not None, snap is not None, version))
        ctx.py = self._py_state(ctx)
        ctx.args_spec = (spec, tuple(id(t) for t in flat))
        key = (tuple(mods), spec, ctx.py, tuple(sig), torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
               torch.get_default_dtype(), torch.is_autocast_enabled(), torch.are_deterministic_algorithms_enabled())
        hash(key)
        return key, ctx

    def _py_state(self, ctx: _Ctx) -> tuple:
        out = []
        seen = set()
        np_, py_ = False, False
        for m in ctx.rel_mod.values():
            tm = type(m)
            if id(tm) in seen:
                continue
            seen.add(id(tm))
            refs = _class_refs(tm, self._refs)
            np_ = np_ or refs.rng_np
            py_ = py_ or refs.rng_py
            out.append(_refs_state(refs))
        ctx.rng_flags = (np_, py_)
        return tuple(out)

    # ── recording a miss ──────────────────────────────────────────────────────
    def begin(self, key, mod, name, ctx, frame) -> Recording:
        rec, values = self.rec, self.values
        r = Recording()
        r.key, r.mod, r.name, r.ctx, r.frame = key, mod, name, ctx, id(frame)
        r.failed = None
        r.start = len(self.log)
        r.root = rec.stack[-1]
        r.nv0, r.no0, r.ni0 = len(values.values), len(rec.data["operations"]), len(rec.data["invocations"])
        r.seq0, r.bytes0, r.bb0 = rec.sequence, rec.bytes, rec.binding_bytes
        r.nb0 = len(rec.data["sourceBindings"])
        r.pc0 = rec.parameter_counter.remaining
        r.obj0, r.sto0 = values.counters.get("object", 0), values.counters.get("storage", 0)
        r.nmod0, r.ndef0, r.ndiag0 = len(rec.data["modules"]), len(rec.data["definitions"]), len(rec.data["diagnostics"])
        cov = rec.data["coverage"]
        r.cov0 = (cov["operationsSeen"], cov["invocationsSeen"])
        prior_s, prior_b = self.suppress, rec.busy
        self.suppress, rec.busy = True, True
        try:
            r.slot_pre = [self.slot_state(t) for t in ctx.slots]
            r.epochs_pre = ([values.write_epochs.get(p[0], 0) if p[0] else 0 for p in r.slot_pre],
                            [values.storage_epochs.get(p[1], 0) if p[1] else 0 for p in r.slot_pre])
            snap = []
            seen = set()
            for m in ctx.rel_mod.values():
                if id(m) in seen:
                    continue
                seen.add(id(m))
                d = m.__dict__
                deep = {k: (list(v) if type(v) is list else dict(v) if type(v) is dict else
                            set(v) if type(v) is set else None) for k, v in d.items()
                        if type(v) in (list, dict, set) and k not in _SKIP_ATTRS}
                snap.append((m, dict(d), deep, dict(m._parameters), dict(m._buffers), dict(m._modules)))
            r.snap = snap
            use_np, use_py = ctx.rng_flags
            np_state = None
            if use_np and "numpy" in sys.modules:
                np_state = sys.modules["numpy"].random.get_state()
            r.rng = (random.getstate() if use_py else None, np_state, torch.initial_seed(), _torch_rng_bytes())
            r.ext_storages = set(ctx.slot_storage)
        finally:
            self.suppress, rec.busy = prior_s, prior_b
        self.active.append(r)
        self.boundaries[id(frame)] = r
        return r

    def fail_all(self, why: str) -> None:
        for r in self.active:
            if r.failed is None:
                r.failed = why

    def _close(self, r: Recording) -> None:
        if self.active and self.active[-1] is r:
            self.active.pop()
        elif r in self.active:
            self.active.remove(r)
        self.boundaries.pop(r.frame, None)
        # Journal entries of enclosing recordings may still name this one; drop
        # everything that references live objects.
        r.args = r.ctx = r.snap = r.slot_pre = r.mod = r.root = None
        if not self.active:
            self.log.clear()

    def abort(self, r: Recording, why: str) -> None:
        self._close(r)
        self.stats["inelig:" + why] += 1
        self.stats["misses"] += 1
        self.bad_keys[r.key] = why

    def end(self, r: Recording, output: Any) -> None:
        rec = self.rec
        # The last operation's result is pending until the next recorder event.
        # Flushing now is what the following leave() would do first anyway.
        rec.guard(rec.flush)
        prior_s, prior_b = self.suppress, rec.busy
        self.suppress, rec.busy = True, True
        why = r.failed
        tpl = None
        try:
            if why is None:
                why = self._side_effects(r)
            if why is None:
                try:
                    tpl = self._build(r, output)
                except Ineligible as e:
                    why = str(e)
                except Exception as e:  # noqa: BLE001
                    why = "internal_error:" + type(e).__name__
        except Exception as e:  # noqa: BLE001
            why = "internal_error:" + type(e).__name__
        finally:
            self.suppress, rec.busy = prior_s, prior_b
            self._close(r)
        self.stats["misses"] += 1
        if why is not None:
            self.stats["inelig:" + why] += 1
            self.bad_keys[r.key] = why
        else:
            self.templates[r.key] = tpl
            self.stats["templates"] += 1

    def _side_effects(self, r: Recording) -> Optional[str]:
        for m, d0, deep, ps, bs, ms in r.snap:
            d = m.__dict__
            if d.keys() != d0.keys():
                return "attr_write"
            for k, v in d.items():
                v0 = d0[k]
                if v is not v0:
                    if type(v) in _PRIM and type(v) is type(v0) and v == v0:
                        continue
                    return "attr_write"
            for k, v0 in deep.items():
                v = d[k]
                if type(v0) is list and v != v0 or type(v0) is set and v != v0:
                    return "attr_write"
                if type(v0) is dict and (v.keys() != v0.keys() or any(v[x] is not v0[x] for x in v0)):
                    return "attr_write"
            for cur, old in ((m._parameters, ps), (m._buffers, bs), (m._modules, ms)):
                if cur.keys() != old.keys() or any(cur[x] is not old[x] for x in old):
                    return "attr_write"
        ctx = r.ctx
        if self._py_state(ctx) != ctx.py:
            return "py_state_write"
        try:
            again: List[torch.Tensor] = []
            spec2 = (_flatten(r.args[0], again), _flatten(r.args[1], again))
        except Ineligible:
            return "arg_container_write"
        if (spec2, tuple(id(t) for t in again)) != ctx.args_spec:
            return "arg_container_write"
        use_np, use_py = ctx.rng_flags
        np_state = None
        if use_np and "numpy" in sys.modules:
            np_state = sys.modules["numpy"].random.get_state()
        st, np0, seed, trng = r.rng
        if trng != _torch_rng_bytes():
            return "rng_torch_state"
        if use_py and random.getstate() != st:
            return "rng_python"
        if torch.initial_seed() != seed:
            return "rng_torch_seed"
        if (np0 is None) != (np_state is None):
            return "rng_numpy"
        if np0 is not None and (np0[1].tobytes() != np_state[1].tobytes() or np0[2:] != np_state[2:]):
            return "rng_numpy"
        return None

    def _build(self, r: Recording, output: Any) -> Template:
        rec, values = self.rec, self.values
        data = rec.data
        ctx = r.ctx
        if rec.stopped or rec.pending is not None or rec.writes:
            raise Ineligible("recorder_state")
        if (len(data["modules"]), len(data["definitions"]), len(data["diagnostics"])) != (r.nmod0, r.ndef0, r.ndiag0):
            raise Ineligible("recorder_registry")
        if not rec.stack or rec.stack[-1] is not r.root:
            raise Ineligible("recorder_stack")
        events = self.log[r.start:]
        facts_new = values.values[r.nv0:]
        v_events = [e for e in events if e[0] == "V"]
        if len(v_events) != len(facts_new) or any(e[1] is not f for e, f in zip(v_events, facts_new)):
            raise Ineligible("fact_journal")
        # Slot state at begin; slots whose pre-existing identities seed the maps.
        slot_of_id: Dict[int, int] = {}
        for i, t in enumerate(ctx.slots):
            slot_of_id.setdefault(id(t), i)
        tid_sym: Dict[str, Any] = {}
        oid_sym: Dict[str, Any] = {}
        sid_sym: Dict[str, Any] = {}
        for i, (oid, sid, snap, _v) in enumerate(r.slot_pre):
            if snap:
                tid_sym.setdefault(snap, ("s", i))
            if oid:
                oid_sym.setdefault(oid, ("s", i))
            if sid:
                sid_sym.setdefault(sid, ("s", i))
        # External writes: slot epochs must be unchanged.
        for i, (oid, sid, _snap, _v) in enumerate(r.slot_pre):
            if oid and values.write_epochs.get(oid, 0) != r.epochs_pre[0][i]:
                raise Ineligible("write_external")
            if sid and values.storage_epochs.get(sid, 0) != r.epochs_pre[1][i]:
                raise Ineligible("write_external")
        facts: List[_Fact] = []
        ext_bytes = 0
        int_bytes_obs = 0
        internal_oids, internal_sids = set(), set()
        for i, (ev, fact) in enumerate(zip(v_events, facts_new)):
            _, _, tensor_id, wepoch, sepoch, nbytes = ev
            f = _Fact()
            slot = slot_of_id.get(tensor_id)
            oid, sid = fact["objectId"], fact["storageId"]
            f.oid_mint = oid not in oid_sym and int(oid[6:]) >= r.obj0
            f.sid_mint = sid not in sid_sym and int(sid[7:]) >= r.sto0
            if not f.oid_mint and oid not in oid_sym:
                raise Ineligible("unmapped_object")
            if not f.sid_mint and sid not in sid_sym:
                raise Ineligible("unmapped_storage")
            if f.oid_mint:
                oid_sym[oid] = ("f", i)
            if f.sid_mint:
                sid_sym[sid] = ("f", i)
            tid_sym[fact["id"]] = ("f", i)
            if slot is not None:
                f.ext = slot
                ext_bytes += nbytes
            else:
                f.ext = None
                if fact["producer"] is None or fact["origin"] != "operation" or fact["state"]:
                    raise Ineligible("external_tensor")
                if oid_sym[oid][0] == "s":
                    raise Ineligible("internal_fact_on_slot_object")
                f.oid, f.sid = oid_sym[oid], sid_sym[sid]
                internal_oids.add(oid)
                if f.sid_mint or (sid in internal_sids):
                    internal_sids.add(sid)
                f.static = {k: fact[k] for k in ("version", "shape", "dtype", "device", "requiresGrad",
                                                 "stride", "storageOffset", "logicalBytes")}
                f.wepoch = wepoch
                f.sepoch = sepoch if sid in internal_sids else None
                p = fact["producer"]
                op_index = self._op_index(p["operationId"], r)
                f.producer = (op_index, p["kind"], p["path"])
                f.prod_fixed = _size(p) - _jlen(p["operationId"])
                initial = dict(fact, producer=None, origin="external")
                toks = [initial["id"], initial["objectId"], initial["storageId"], initial["aliasOf"],
                        initial["previousVersion"], initial["logicalVersion"]]
                f.fixed = _size(initial) - sum(_jlen(x) for x in toks)
                if nbytes != f.fixed + sum(_jlen(x) for x in toks) + 8:
                    raise Ineligible("byte_model_fact")
                int_bytes_obs += nbytes
            facts.append(f)
        # Epoch finals for internal objects/storages.
        epochs = ([(oid_sym[o], values.write_epochs[o]) for o in internal_oids if o in values.write_epochs],
                  [(sid_sym[s], values.storage_epochs[s]) for s in internal_sids if s in values.storage_epochs])
        # Records: nested invocations and operations, merged by sequence.
        invs = data["invocations"][r.ni0:]
        ops = data["operations"][r.no0:]
        root_id = r.root["id"]
        root_seq = r.root["sequence"]
        inv_index = {inv["id"]: k for k, inv in enumerate(invs)}
        binds = [e for e in events if e[0] == "B"]
        merged = sorted([("i", k, inv["sequence"]) for k, inv in enumerate(invs)] +
                        [("o", k, op["sequence"]) for k, op in enumerate(ops)], key=lambda x: x[2])
        if rec.sources is not None and len(binds) != len(merged):
            raise Ineligible("binding_journal")
        u_depth = -1
        bind_syms = []
        for e in binds:
            _, found, passed, exhausted = e
            mine = None
            for pr, dist in passed or ():
                if pr is r:
                    mine = dist
            if mine is None:
                bind_syms.append(("I", found))
            else:
                if exhausted:
                    raise Ineligible("frame_depth")
                bind_syms.append(("U", None))
                u_depth = max(u_depth, mine)

        def tid(x):
            try:
                return _V(("t", tid_sym[x]))
            except KeyError:
                raise Ineligible("unmapped_value") from None

        def tree(node):
            if type(node) is dict:
                if node.get("kind") == "value":
                    return {k: (tid(v) if k == "id" else v) for k, v in node.items()}
                if node.get("kind") == "opaque":
                    raise Ineligible("opaque_record")
                return {k: tree(v) for k, v in node.items()}
            if type(node) is list:
                return [tree(v) for v in node]
            return node

        def seq(s):
            if s is None:
                return None
            return _V(("q", s - root_seq))

        def inv_ref(x):
            if x == root_id:
                return _V(("root",))
            if x in inv_index:
                return _V(("i", inv_index[x]))
            raise Ineligible("foreign_invocation")

        records = []
        total = 0
        tpl_resolve = _TplResolver(self, r, invs, ops, tid_sym)
        for j, (kind, k, _s) in enumerate(merged):
            bsym = bind_syms[j] if rec.sources is not None else None
            if kind == "i":
                inv = invs[k]
                if inv["status"] != "completed" or inv["diagnosticId"] is not None:
                    raise Ineligible("invocation_status")
                mod_rel = self._module_rel(inv["moduleId"], r)
                base = dict(inv, id=_V(("i", k)), moduleId=_V(("m", mod_rel)), parentId=inv_ref(inv["parentId"]),
                            sequence=seq(inv["sequence"]), endSequence=None, arguments=tree(inv["arguments"]),
                            output=None, status="interrupted", sourceBindingIds=_V(("b",)))
                out_tree = _freeze(tree(inv["output"]))
                skel = _freeze(base)
                fixed0 = _fixed(skel, tpl_resolve.bind(bsym, inv["sourceBindingIds"]))
                fixed1 = _fixed(out_tree, tpl_resolve.bind(bsym, inv["sourceBindingIds"]))
                total += fixed0 + 128 + fixed1 + 8 + self._tok(skel, tpl_resolve, bsym, inv["sourceBindingIds"]) \
                    + self._tok(out_tree, tpl_resolve, bsym, inv["sourceBindingIds"])
                records.append(("i", k, skel, out_tree, seq(inv["endSequence"]).sym, bsym, fixed0, fixed1))
            else:
                op = ops[k]
                if op["status"] != "completed" or op["diagnosticId"] is not None:
                    raise Ineligible("operation_status")
                base = dict(op, id=_V(("o", k)), sequence=seq(op["sequence"]),
                            invocationId=inv_ref(op["invocationId"]) if op["invocationId"] is not None else None,
                            arguments=tree(op["arguments"]), output=None, effects=[], status="interrupted",
                            sourceBindingIds=_V(("b",)))
                if op["invocationId"] is None:
                    raise Ineligible("orphan_operation")
                skel = _freeze(base)
                out_tree = _freeze(tree(op["output"]))
                effects = []
                for e in op["effects"]:
                    if e["after"].get("kind") != "value":
                        raise Ineligible("opaque_effect")
                    effects.append(_freeze(dict(e, beforeId=tid(e["beforeId"]), after=tree(e["after"]))))
                rb = tpl_resolve.bind(bsym, op["sourceBindingIds"])
                fixed0 = _fixed(skel, rb)
                fixed1 = _fixed(out_tree, rb)
                fx_eff = [_fixed(e, rb) for e in effects]
                # Producers charged with this operation's output/effects.
                prod_sizes = sum(f.prod_fixed + _jlen(op["id"]) for f in facts if f.ext is None and f.producer[0] == k)
                total += (fixed0 + 128 + self._tok(skel, tpl_resolve, bsym, op["sourceBindingIds"])
                          + fixed1 + 8 + self._tok(out_tree, tpl_resolve, bsym, op["sourceBindingIds"])
                          + sum(fx + 8 + self._tok(e, tpl_resolve, bsym, op["sourceBindingIds"])
                                for fx, e in zip(fx_eff, effects))
                          + prod_sizes)
                records.append(("o", k, skel, out_tree, effects, bsym, fixed0, fixed1, fx_eff))
        # Byte model self-check against the recorder's own accounting.
        observed = rec.bytes - r.bytes0 - (rec.binding_bytes - r.bb0) - ext_bytes
        predicted = total + int_bytes_obs
        if observed != predicted:
            raise Ineligible("byte_model")
        # Legacy (prov/result) bookkeeping of nested calls.
        ev_out = []
        for e in events:
            if e[0] == "E":
                rel = ctx.mod_rel.get(id(e[1]))
                if not rel:
                    raise Ineligible("foreign_module")
                ev_out.append(("E", rel, e[2]))
            elif e[0] == "X":
                rel = ctx.mod_rel.get(id(e[1]))
                if not rel:
                    raise Ineligible("foreign_module")
                ev_out.append(("X", rel, e[2], e[3]))
        # Outputs.
        outs: List[torch.Tensor] = []
        try:
            out_spec = _flatten(output, outs)
        except Ineligible as e:
            raise Ineligible("output_" + str(e))
        out_slots = []
        seen: Dict[int, int] = {}
        storages: Dict[int, int] = {}  # storage -> id() of the root tensor fabricated for it

        def need(x):
            return 0 if x.numel() == 0 else (1 + sum((d - 1) * s for d, s in zip(x.shape, x.stride()))) * x.element_size()

        def internal(x, st):
            oid = weak_get(values.objects, x)
            sid = weak_get(values.storages, st)
            if sid is None or sid not in internal_sids or (oid is not None and oid not in internal_oids):
                raise Ineligible("external_output")
            if x.requires_grad or isinstance(x, nn.Parameter):
                raise Ineligible("output_kind")
            return oid, sid

        w_events = [e for e in events if e[0] == "W"]
        expected_alive = set()
        for j, t in enumerate(outs):
            if id(t) in seen:
                out_slots.append(("dup", seen[id(t)]))
                continue
            seen[id(t)] = j
            i = slot_of_id.get(id(t))
            if i is not None:
                out_slots.append(("slot", i))
                continue
            st = t.untyped_storage()
            sc = st._cdata
            if sc in r.ext_storages:
                raise Ineligible("output_aliases_input")
            base = t._base
            root = t if base is None else base
            if base is not None and (base._base is not None or base.dtype != t.dtype):
                raise Ineligible("output_view")
            if root.storage_offset() != 0 or st.nbytes() != need(root):
                raise Ineligible("output_storage")
            if storages.setdefault(sc, id(root)) != id(root):
                raise Ineligible("outputs_share_storage")
            oid, sid = internal(t, st)
            if oid is None:
                raise Ineligible("external_output")
            root_oid = oid if base is None else internal(base, st)[0]
            root_meta = (tuple(root.shape), root.stride(), root.dtype, root.device, _version(root))
            chain = []
            if base is not None:
                # Replay the same view operations from the fabricated root: under
                # FakeTensorMode each view keeps its parent view alive, and those
                # parents stay registered with the recorder exactly as live ones do.
                cur = t
                while cur is not root:
                    step = next((w for w in reversed(w_events) if id(cur) in w[5]), None)
                    if step is None:
                        raise Ineligible("view_chain")
                    entry = values.objects.get(step[4])
                    parent = entry[0]() if entry is not None else None
                    if parent is None or len(chain) > 64:
                        raise Ineligible("view_chain")
                    cur_oid = weak_get(values.objects, cur)
                    chain.append((step[1], step[2], step[3], step[5].index(id(cur)) if step[6] else None,
                                  None if cur_oid is None else oid_sym.get(cur_oid), id(cur),
                                  (tuple(cur.shape), cur.stride(), cur.storage_offset())))
                    if cur_oid is not None:
                        if cur_oid not in internal_oids:
                            raise Ineligible("external_output")
                        expected_alive.add(cur_oid)
                    cur = parent
                chain.reverse()
            expected_alive.add(oid)
            if root_oid is not None:
                expected_alive.add(root_oid)
            out_slots.append(("new", oid_sym[oid], None if root_oid is None else oid_sym[root_oid], sid_sym[sid],
                              root_meta, tuple(chain), id(root)))
        # Every internal object still alive now must be one the stamp recreates.
        alive = set()
        for o in internal_oids:
            s = values.object_storage.get(o)
            ref = values.storage_objects.get(s, {}).get(o) if s is not None else None
            if ref is not None and ref() is not None:
                alive.add(o)
        if alive != expected_alive:
            raise Ineligible("live_interior")
        tpl = Template()
        tpl.facts = facts
        tpl.records = records
        tpl.events = ev_out
        tpl.out_spec = out_spec
        tpl.out_slots = out_slots
        tpl.n_facts = len(facts)
        tpl.n_ops = len(ops)
        tpl.n_inv = len(invs)
        tpl.ticks = rec.sequence - r.seq0
        tpl.ops_seen = data["coverage"]["operationsSeen"] - r.cov0[0]
        tpl.inv_seen = data["coverage"]["invocationsSeen"] - r.cov0[1]
        tpl.pc_delta = r.pc0 - rec.parameter_counter.remaining
        tpl.obj_delta = values.counters.get("object", 0) - r.obj0
        tpl.sto_delta = values.counters.get("storage", 0) - r.sto0
        tpl.bytes_tpl = rec.bytes - r.bytes0
        tpl.n_tokens = 8 * len(facts) + 32 * (len(ops) + len(invs))
        tpl.u_depth = u_depth
        tpl.has_u = u_depth >= 0
        tpl.epochs = epochs
        tpl.rels = sorted({x[1] for x in ev_out} | {self._module_rel(inv["moduleId"], r) for inv in invs})
        tpl.cls = type(r.mod).__name__
        if (tpl.ops_seen, tpl.inv_seen) != (len(ops), len(invs)):
            raise Ineligible("coverage")
        return tpl

    def _tok(self, node, resolver, bsym, tpl_bind_ids) -> int:
        acc = [0]
        _inst(node, resolver.bind(bsym, tpl_bind_ids), acc)
        return acc[0]

    def _op_index(self, op_id: str, r: Recording) -> int:
        k = int(op_id.split(":")[1]) - r.no0
        if k < 0:
            raise Ineligible("foreign_operation")
        return k

    def _module_rel(self, module_id: str, r: Recording) -> str:
        rec = self.rec
        for rel, m in r.ctx.rel_mod.items():
            found = weak_get(rec.instances, m)
            if found is not None and found["id"] == module_id:
                if not rel:
                    raise Ineligible("recursive_root")
                return rel
        raise Ineligible("foreign_module")

    # ── hooks called from trace.py while recordings are active ───────────────
    def _op_info(self, func: Any) -> tuple:
        try:
            tags = func.tags
        except Exception:
            tags = ()
        seeded = torch.Tag.nondeterministic_seeded in tags
        bad = None
        for tag in (torch.Tag.data_dependent_output, torch.Tag.dynamic_output_shape):
            if tag in tags:
                bad = "value_read:" + str(func)
                break
        name = str(func)
        if bad is None and any(v in name for v in _VALUE_READS):
            bad = "value_read"
        try:
            writes = tuple((i, a.name) for i, a in enumerate(func._schema.arguments)
                           if a.alias_info is not None and a.alias_info.is_write)
            view = not writes and any(r.alias_info is not None for r in func._schema.returns)
        except Exception:
            writes, view = (), False
            bad = bad or "unknown_schema"
        info = (seeded, bad, writes, view)
        self._ops[func] = info
        return info

    def on_dispatch(self, func: Any, args: tuple, kwargs: dict) -> None:
        rec = self.rec
        prior = rec.busy
        rec.busy = True  # tensor introspection here is not the model's
        try:
            self._on_dispatch(func, args, kwargs)
        except Exception as e:  # noqa: BLE001
            self.stats["internal_error:dispatch:" + type(e).__name__] += 1
            self.fail_all("internal_error")
        finally:
            rec.busy = prior

    def _on_dispatch(self, func: Any, args: tuple, kwargs: dict) -> None:
        info = self._ops.get(func)
        if info is None:
            info = self._op_info(func)
        seeded, bad, writes, _view = info
        if seeded and not _rng_unused(func, args, kwargs):
            self.fail_all("rng_op:" + str(func))
        if bad is not None and not (func is _INDEX_TENSOR and _integer_indices(args)):
            self.fail_all(bad)
        if writes:
            for i, an in writes:
                v = args[i] if i < len(args) else kwargs.get(an)
                for t in _tensors(v):
                    self._check_write(t)

    def on_result(self, func: Any, args: tuple, kwargs: dict, output: Any) -> None:
        """Journal view operations: a fake view keeps its parent alive (see _build)."""
        info = self._ops.get(func) or self._op_info(func)
        if not info[3]:
            return
        parents = []

        def spec(v):
            if isinstance(v, torch.Tensor):
                parents.append(id(v))
                return _PARENT
            if type(v) in (list, tuple):
                return type(v)(spec(x) for x in v)
            if type(v) in _PRIM:
                return v
            raise Ineligible("view_argument")
        try:
            a, k = tuple(spec(v) for v in args), {n: spec(v) for n, v in kwargs.items()}
        except Ineligible:
            return
        if len(parents) != 1:
            return
        self.log.append(("W", func, a, k, parents[0], [id(t) for t in _tensors(output)],
                         isinstance(output, (list, tuple))))

    def _check_write(self, t: torch.Tensor) -> None:
        try:
            st = t.untyped_storage()
            sc = st._cdata
            sid = weak_get(self.values.storages, st)
        except Exception:
            sc, sid = None, None
        for r in self.active:
            if r.failed is None and (sc is None or sid is None or sc in r.ext_storages or int(sid[7:]) < r.sto0):
                r.failed = "write_external"

    def log_enter(self, mod: nn.Module, in_shape: Any) -> None:
        self.log.append(("E", mod, in_shape))

    def log_exit(self, mod: nn.Module, out_shape: Any, output: Any) -> None:
        meta = None
        if isinstance(output, torch.Tensor):
            meta = {"shape": list(output.shape), "dtype": str(output.dtype).removeprefix("torch."),
                    "logicalBytes": output.numel() * output.element_size()}
        self.log.append(("X", mod, out_shape, meta))

    # ── stamping a hit ────────────────────────────────────────────────────────
    def precheck(self, tpl: Template, ctx: _Ctx, call_frame: Any):
        rec, values = self.rec, self.values
        lim = rec.limits
        data = rec.data
        if rec.stopped or rec.finished or rec.pending is not None or not rec.stack or rec.stack[-1] is None:
            return "recorder_state"
        if len(data["invocations"]) + tpl.n_inv + 1 >= lim.invocations:
            return "invocation_headroom"
        if len(data["operations"]) + tpl.n_ops + 1 >= lim.operations:
            return "operation_headroom"
        if len(values.values) + tpl.n_facts + 1 >= lim.values:
            return "value_headroom"
        if rec.bytes + tpl.bytes_tpl + 8 * tpl.n_tokens + 8192 > lim.bytes:
            return "byte_headroom"
        if rec.parameter_counter.remaining < tpl.pc_delta + 1:
            return "parameter_budget"
        modules = {}
        for rel in tpl.rels:
            m = ctx.rel_mod.get(rel)
            found = weak_get(rec.instances, m) if m is not None else None
            n = self.names.get(id(m)) if m is not None else None
            if found is None or n is None:
                return "module_unresolved"
            modules[rel] = (found["id"], n)
        u = None
        if tpl.has_u and rec.sources is not None:
            u = self._outside(call_frame, tpl.u_depth)
            if u is _NO:
                return "frame_depth"
        return modules, u

    def _outside(self, frame: Any, depth: int):
        """The binding an operation inside the call would find outside it."""
        from ..semantic import source_bindings
        sources = self.rec.sources
        internal, torch_dir = sources.internal, sources.torch_directory
        try:
            for j in range(source_bindings.MAX_FRAMES):
                if frame is None:
                    break
                if depth + j >= source_bindings.MAX_FRAMES:
                    return _NO
                filename = frame.f_code.co_filename
                if filename not in internal and not filename.startswith(torch_dir):
                    return ("found", frame.f_code, frame.f_lasti)
                frame = frame.f_back
        finally:
            del frame
        return _NO

    def stamp(self, tpl: Template, name: str, mod: nn.Module, ctx: _Ctx, call_frame: Any) -> Any:
        rec = self.rec
        pre = self.precheck(tpl, ctx, call_frame)
        if type(pre) is str:
            self.stats["precheck:" + pre] += 1
            return _NO_STAMP
        modules, u = pre
        try:
            with rec.mute():
                outs = self._apply(tpl, ctx, modules, u)
            output = self._fabricate(tpl, ctx, outs)
        except Exception as e:  # noqa: BLE001
            # Pre-checks make this unreachable. A half-written capture must never be
            # published as complete: stop the recorder (a recorder_error capture, as
            # for any recorder failure) and let the forward itself continue live.
            self.stats["internal_error:stamp:" + (str(e) if isinstance(e, _StampError) else type(e).__name__)] += 1
            rec.omit("recorder_error")
            return _NO_STAMP
        self.stats["hits"] += 1
        self.stats["stamped_ops"] += tpl.n_ops
        return output

    def _apply(self, tpl: Template, ctx: _Ctx, modules: Dict[str, Tuple[str, str]], u: Any):
        rec, values = self.rec, self.values
        data = rec.data
        vals = values.values
        counters = values.counters
        root = rec.stack[-1]
        root_id = root["id"]
        root_seq = root["sequence"]
        slots = ctx.slots
        slot_now = [None] * len(slots)

        def slot(i):
            s = slot_now[i]
            if s is None:
                s = slot_now[i] = self.slot_state(slots[i])
            return s

        f_tid: List[Optional[str]] = [None] * tpl.n_facts
        f_oid: Dict[int, str] = {}
        f_sid: Dict[int, str] = {}

        def oid_of(sym):
            return slot(sym[1])[0] if sym[0] == "s" else f_oid[sym[1]]

        def sid_of(sym):
            return slot(sym[1])[1] if sym[0] == "s" else f_sid[sym[1]]

        latest, storage_latest, by_id = values.latest, values.storage_latest, values.by_id
        snapshots, metadata, snapshot_epochs = values.snapshots, values.metadata, values.snapshot_epochs
        observe = self._orig_observe
        op0 = len(data["operations"])
        inv0 = len(data["invocations"])
        obj0, sto0 = counters.get("object", 0), counters.get("storage", 0)
        new_facts = []
        nbytes = 0
        for i, f in enumerate(tpl.facts):
            if f.ext is not None:
                n0 = len(vals)
                fact = observe(slots[f.ext])
                if fact is None or len(vals) != n0 + 1:
                    raise _StampError("external_observe")
                f_tid[i] = fact["id"]
                if f.oid_mint:
                    f_oid[i] = fact["objectId"]
                if f.sid_mint:
                    f_sid[i] = fact["storageId"]
                new_facts.append(None)
                continue
            if f.oid_mint:
                n = counters.get("object", 0)
                counters["object"] = n + 1
                oid = f_oid[i] = f"object{n}"
            else:
                oid = oid_of(f.oid)
            if f.sid_mint:
                n = counters.get("storage", 0)
                counters["storage"] = n + 1
                sid = f_sid[i] = f"storage{n}"
            else:
                sid = sid_of(f.sid)
            tid = f"tensor{len(vals)}"
            f_tid[i] = tid
            st = f.static
            prev = latest.get(oid)
            alias = storage_latest.get(sid)
            previous = by_id.get(prev)
            fact = dict(id=tid, objectId=oid, storageId=sid, version=st["version"], shape=st["shape"],
                        dtype=st["dtype"], device=st["device"], requiresGrad=st["requiresGrad"],
                        aliasOf=alias if alias != prev else None,
                        previousVersion=previous["id"] if previous is not None else None,
                        logicalVersion=previous["logicalVersion"] + 1 if previous is not None else 0,
                        stride=st["stride"], storageOffset=st["storageOffset"], logicalBytes=st["logicalBytes"],
                        state=[], producer=None, origin="operation")
            nbytes += (f.fixed + _jlen(tid) + _jlen(oid) + _jlen(sid) + _jlen(fact["aliasOf"])
                       + _jlen(fact["previousVersion"]) + _jlen(fact["logicalVersion"]) + 8)
            vals.append(fact)
            key = (oid, st["version"], f.wepoch)
            snapshots[key] = fact
            metadata[key] = (st["shape"], st["stride"], st["storageOffset"], st["requiresGrad"], sid)
            snapshot_epochs[key] = f.sepoch if f.sepoch is not None else values.storage_epochs.get(sid, 0)
            latest[oid] = storage_latest[sid] = tid
            by_id[tid] = fact
            new_facts.append(fact)
        for (sym, e) in tpl.epochs[0]:
            values.write_epochs[oid_of(sym)] = e
        for (sym, e) in tpl.epochs[1]:
            values.storage_epochs[sid_of(sym)] = e
        for i, f in enumerate(tpl.facts):
            if f.ext is None:
                k, kind, path = f.producer
                op_id = f"operation:{op0 + k}"
                new_facts[i]["producer"] = dict(operationId=op_id, kind=kind, path=path)
                nbytes += f.prod_fixed + _jlen(op_id)

        def tid_of(sym):
            return slot(sym[1])[2] if sym[0] == "s" else f_tid[sym[1]]

        u_binding = None
        if u is not None and u is not _NO:
            u_binding = rec.sources.expression(u[1], u[2])
        bind_ids: List[Any] = [None]

        def resolve(sym):
            kind = sym[0]
            if kind == "t":
                return tid_of(sym[1])
            if kind == "q":
                return root_seq + sym[1]
            if kind == "o":
                return f"operation:{op0 + sym[1]}"
            if kind == "i":
                return f"invocation:{inv0 + sym[1]}"
            if kind == "root":
                return root_id
            if kind == "m":
                return modules[sym[1]][0]
            if kind == "b":
                return bind_ids[0]
            raise _StampError("symbol")

        invocations, operations = data["invocations"], data["operations"]
        for item in tpl.records:
            bsym = item[5]
            if bsym is None:
                bind_ids[0] = []
            else:
                b = bsym[1] if bsym[0] == "I" else u_binding
                bind_ids[0] = rec.source_binding(b)
            acc = [0]
            if item[0] == "i":
                _, k, skel, out_tree, end_q, _b, fixed0, fixed1 = item
                record = _inst(skel, resolve, acc)
                b0 = fixed0 + 128 + acc[0]
                acc2 = [0]
                out = _inst(out_tree, resolve, acc2)
                record.update(endSequence=root_seq + end_q[1], status="completed", output=out)
                nbytes += b0 + fixed1 + 8 + acc2[0]
                invocations.append(record)
            else:
                _, k, skel, out_tree, effects, _b, fixed0, fixed1, fx_eff = item
                record = _inst(skel, resolve, acc)
                b0 = fixed0 + 128 + acc[0]
                acc2 = [0]
                out = _inst(out_tree, resolve, acc2)
                effs = []
                eb = 0
                for fx, e in zip(fx_eff, effects):
                    acc3 = [0]
                    effs.append(_inst(e, resolve, acc3))
                    eb += fx + 8 + acc3[0]
                record.update(output=out, effects=effs, status="completed")
                nbytes += b0 + fixed1 + 8 + acc2[0] + eb
                operations.append(record)
        rec.bytes += nbytes
        rec.sequence = root_seq + 1 + tpl.ticks
        cov = data["coverage"]
        cov["operationsSeen"] += tpl.ops_seen
        cov["invocationsSeen"] += tpl.inv_seen
        rec.parameter_counter.remaining -= tpl.pc_delta
        values.reason = None
        # Legacy bookkeeping of nested calls (trace.py _install_hooks.call).
        result = self.result
        stack = []
        for ev in tpl.events:
            n = modules[ev[1]][1]
            if ev[0] == "E":
                result.visited.add(n)
                result.call_counts[n] = result.call_counts.get(n, 0) + 1
                rio = result.io.setdefault(n, {"in": None, "out": None})
                rio["out"] = None
                rio.pop("output_tensor", None)
                if rio["in"] is None:
                    rio["in"] = None if ev[2] is None else list(ev[2])
                stack.append(n)
            else:
                n = stack.pop()
                rio = result.io[n]
                rio["out"] = None if ev[2] is None else list(ev[2])
                meta = ev[3]
                if meta is not None and result.call_counts[n] == 1:
                    rio["output_tensor"] = dict(meta, shape=list(meta["shape"]))
        if (counters.get("object", 0), counters.get("storage", 0)) != (obj0 + tpl.obj_delta, sto0 + tpl.sto_delta):
            raise _StampError("counters")
        outs = []
        for sl in tpl.out_slots:
            if sl[0] == "new":
                outs.append((oid_of(sl[1]), None if sl[2] is None else oid_of(sl[2]), sid_of(sl[3]),
                             [None if step[4] is None else oid_of(step[4]) for step in sl[5]]))
            else:
                outs.append(None)
        return outs

    def _fabricate(self, tpl: Template, ctx: _Ctx, outs: List[Any]) -> Any:
        rec = self.rec
        objs = []
        roots: Dict[int, torch.Tensor] = {}
        prior_s, prior_b = self.suppress, rec.busy
        self.suppress, rec.busy = True, True
        try:
            for sl, ids in zip(tpl.out_slots, outs):
                kind = sl[0]
                if kind == "slot":
                    objs.append(ctx.slots[sl[1]])
                elif kind == "dup":
                    objs.append(objs[sl[1]])
                else:
                    _, _o, _r, _s, root_meta, chain, root_key = sl
                    oid, root_oid, sid, chain_oids = ids
                    root = roots.get(root_key)
                    if root is None:
                        shape, stride, dtype, device, version = root_meta
                        root = roots[root_key] = torch.empty_strided(shape, stride, dtype=dtype, device=device)
                        if version:
                            torch._C._autograd._unsafe_set_version_counter((root,), (version,))
                        if root_oid is not None:
                            self._register(root, root_oid, sid)
                    t = root
                    for (func, a, k, index, _sym, key, meta), step_oid in zip(chain, chain_oids):
                        made = roots.get(key)
                        if made is None:
                            made = func(*_bind_parent(a, t), **{n: _bind_parent(v, t) for n, v in k.items()})
                            if index is not None:
                                made = made[index]
                            if (tuple(made.shape), made.stride(), made.storage_offset()) != meta:
                                raise _StampError("view_replay")
                            roots[key] = made
                            if step_oid is not None:
                                self._register(made, step_oid, sid)
                        t = made
                    objs.append(t)
        finally:
            self.suppress, rec.busy = prior_s, prior_b
        return _rebuild(tpl.out_spec, iter(objs))

    def _register(self, tensor: torch.Tensor, oid: str, sid: str) -> None:
        """What observe() leaves behind for a live object: weak identities and its storage bucket."""
        values = self.values
        weak_set(values.objects, tensor, oid)
        weak_set(values.storages, tensor.untyped_storage(), sid)
        values.object_storage[oid] = sid

        def released(reference):
            bucket = values.storage_objects.get(sid, {})
            if bucket.get(oid) is reference:
                del bucket[oid]
                values.object_storage.pop(oid, None)
                if not bucket:
                    values.storage_objects.pop(sid, None)
        values.storage_objects.setdefault(sid, {})[oid] = weakref.ref(tensor, released)

    # ── shadow mode ───────────────────────────────────────────────────────────
    def shadow_stamp(self, tpl: Template, name: str, mod: nn.Module, ctx: _Ctx, call_frame: Any):
        """Stamp and leave() on the live recorder, capture, then restore it."""
        rec = self.rec
        pre = self.precheck(tpl, ctx, call_frame)
        if type(pre) is str:
            self.stats["precheck:" + pre] += 1
            return None
        modules, u = pre
        cp = _Checkpoint(self)
        cp.name = name
        captured = None
        try:
            with rec.mute():
                outs = self._apply(tpl, ctx, modules, u)
            output = self._fabricate(tpl, ctx, outs)
            rec.guard(rec.leave, output)
            prior = rec.busy
            rec.busy = True
            try:
                captured = cp.capture(output)
            finally:
                rec.busy = prior
        except Exception as e:  # noqa: BLE001 - reported as a mismatch
            captured = ("error", repr(e))
        finally:
            cp.restore()
        self.stats["hits"] += 1
        return cp, captured

    def shadow_check(self, shadow: Any, output: Any) -> None:
        cp, stamped = shadow
        prior = self.rec.busy
        self.rec.busy = True
        try:
            live = cp.capture(output)
        except Exception as e:  # noqa: BLE001
            live = ("error", repr(e))
        finally:
            self.rec.busy = prior
        self.stats["shadow_compared"] += 1
        if stamped != live:
            self.stats["shadow_mismatch"] += 1
            if len(self.mismatches) < 8:
                self.mismatches.append(dict(module=cp.name, path=str(_first_diff(stamped, live))[:400]))


class _TplResolver:
    """Resolve skeleton symbols back to the template's own values (for sizing)."""

    def __init__(self, cache, r, invs, ops, tid_sym):
        self.r, self.invs, self.ops = r, invs, ops
        self.rev_t = {v: k for k, v in tid_sym.items()}
        self.root_seq = r.root["sequence"]
        self.cache = cache

    def bind(self, bsym, ids):
        r, invs, ops = self.r, self.invs, self.ops

        def resolve(sym):
            kind = sym[0]
            if kind == "t":
                return self.rev_t[sym[1]]
            if kind == "q":
                return self.root_seq + sym[1]
            if kind == "o":
                return ops[sym[1]]["id"]
            if kind == "i":
                return invs[sym[1]]["id"]
            if kind == "root":
                return r.root["id"]
            if kind == "m":
                return weak_get(self.cache.rec.instances, r.ctx.rel_mod[sym[1]])["id"]
            if kind == "b":
                return ids
            raise Ineligible("symbol")
        return resolve


_NO = object()
_NO_STAMP = object()
_PARENT = ("<parent view>",)


def _bind_parent(v, parent):
    if v is _PARENT:
        return parent
    if type(v) in (list, tuple):
        return type(v)(_bind_parent(x, parent) for x in v)
    return v


def _tensors(value: Any, acc: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
    acc = [] if acc is None else acc
    if isinstance(value, torch.Tensor):
        acc.append(value)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _tensors(v, acc)
    return acc


def _rebuild(spec: Any, it) -> Any:
    head = spec[0]
    if head == "T":
        return next(it)
    if head is tuple:
        return tuple(_rebuild(s, it) for s in spec[1])
    if head is list:
        return [_rebuild(s, it) for s in spec[1]]
    if head is torch.Size:
        return torch.Size(spec[1])
    if head is dict:
        return {k: _rebuild(s, it) for k, s in spec[1]}
    return spec[1]


def _first_diff(a, b, path=""):
    if type(a) is not type(b):
        return f"{path or '<root>'}: {a!r:.120} != {b!r:.120}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}.{k}: {'stamp-only' if k in a else 'live-only'} {a.get(k, b.get(k))!r:.80}"
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d is not None:
                return d
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}[len {len(a)}!={len(b)}]"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d is not None:
                return d
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


class _Checkpoint:
    """Everything a stamp or a live call can change, restorable in place (shadow only)."""

    def __init__(self, cache: TemplateCache) -> None:
        self.cache = cache
        self.name = ""
        rec, values, data = cache.rec, cache.values, cache.rec.data
        names = ("invocations", "operations", "values", "sourceBindings", "diagnostics", "modules", "definitions")
        self.lens = {k: len(data[k]) for k in names}
        self.scalars = (rec.sequence, rec.bytes, rec.binding_bytes, rec.parameter_counter.remaining,
                        rec.stopped, rec.finished, rec.fake, values.reason, dict(data["coverage"]),
                        rec.pending, list(rec.writes), list(rec.stack))
        self.omissions = set(rec.omissions)
        self.bindings = set(rec.bindings)
        self.diag = dict(rec.diagnostics)
        self.tables = {name: dict(getattr(values, name)) for name in _TABLES}
        self.buckets = {sid: dict(b) for sid, b in values.storage_objects.items()}
        self.weak = {name: dict(getattr(values, name)) for name in ("objects", "storages")}
        self.counters = dict(values.counters)
        res = cache.result
        # The call wrapper holds references to its own io records: restore them in place.
        self.legacy = ({k: (v, json.dumps(v)) for k, v in res.io.items()}, dict(res.call_counts), set(res.visited))
        self.root = rec.stack[-1]
        self.root_copy = json.loads(json.dumps(self.root)) if self.root is not None else None

    def capture(self, output: Any):
        cache = self.cache
        rec, values, data = cache.rec, cache.values, cache.rec.data
        new = {k: json.loads(json.dumps(data[k][self.lens[k]:])) for k in self.lens}
        tables = {}
        for name in _TABLES:
            cur, old = getattr(values, name), self.tables[name]
            diff = {}
            for k, v in cur.items():
                if k not in old or old[k] is not v:
                    diff[str(k)] = v["id"] if isinstance(v, dict) and "id" in v else v
            tables[name] = diff
        outs: List[torch.Tensor] = []
        try:
            _flatten(output, outs)
        except Ineligible:
            outs = []
        out_ids = []
        for t in outs:
            out_ids.append((weak_get(values.objects, t), weak_get(values.storages, t.untyped_storage()),
                            tuple(t.shape), tuple(t.stride()), str(t.dtype), t.storage_offset(), _version(t),
                            t.requires_grad, type(t).__name__))
        res = cache.result
        return dict(new=new, tables=tables, outs=out_ids,
                    scalars=[rec.sequence, rec.bytes, rec.binding_bytes, rec.parameter_counter.remaining,
                             rec.stopped, rec.finished, values.reason, dict(data["coverage"]),
                             rec.pending is None, len(rec.writes)],
                    omissions=sorted(rec.omissions), counters=dict(values.counters),
                    root=json.loads(json.dumps(self.root)) if self.root is not None else None,
                    legacy=(json.loads(json.dumps(res.io, sort_keys=True, default=str)), dict(res.call_counts),
                            sorted(res.visited)))

    def restore(self) -> None:
        cache = self.cache
        rec, values, data = cache.rec, cache.values, cache.rec.data
        for k, n in self.lens.items():
            del data[k][n:]
        (rec.sequence, rec.bytes, rec.binding_bytes, rec.parameter_counter.remaining, rec.stopped, rec.finished,
         rec.fake, values.reason, cov, rec.pending, writes, stack) = self.scalars
        data["coverage"].clear()
        data["coverage"].update(cov)
        rec.writes = writes
        rec.stack[:] = stack
        rec.omissions.clear()
        rec.omissions.update(self.omissions)
        rec.bindings.clear()
        rec.bindings.update(self.bindings)
        rec.diagnostics.clear()
        rec.diagnostics.update(self.diag)
        for name, saved in self.tables.items():
            t = getattr(values, name)
            t.clear()
            t.update(saved)
        values.storage_objects.clear()
        values.storage_objects.update({sid: dict(b) for sid, b in self.buckets.items()})
        for name, saved in self.weak.items():
            t = getattr(values, name)
            t.clear()
            t.update(saved)
        values.counters.clear()
        values.counters.update(self.counters)
        if self.root is not None:
            self.root.clear()
            self.root.update(json.loads(json.dumps(self.root_copy)))
        res = cache.result
        io, counts, visited = self.legacy
        res.io.clear()
        for k, (obj, saved) in io.items():
            obj.clear()
            obj.update(json.loads(saved))
            res.io[k] = obj
        res.call_counts.clear()
        res.call_counts.update(counts)
        res.visited.clear()
        res.visited.update(visited)


_TABLES = ("snapshots", "metadata", "snapshot_epochs", "latest", "storage_latest", "by_id", "storage_epochs",
           "write_epochs", "object_storage")


def raise_gc_threshold():
    """Template allocations can push gen-2 over its trigger inside the forward (one
    ~40 ms collection measured on swin_t). Returns the prior threshold to restore."""
    prior = gc.get_threshold()
    gc.set_threshold(prior[0], prior[1], _GC_GEN2)
    return prior

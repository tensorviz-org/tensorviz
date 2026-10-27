"""Bounded, optional constructor-call provenance during approved model setup.

Python's profiler observes initializer entry, not successful completion. Locations
enable navigation; optional scalar parameter snapshots remain separate evidence.
The observer never retains a frame, module, tensor or source body.
"""
from contextlib import contextmanager
import inspect
import sys
import types
import weakref

from .identity import weak_get, weak_set
from . import initializer_arguments

MAX_CONSTRUCTIONS = 8192
_MISSING = object()
_UNAVAILABLE = object()


def _lookup(items, name):
    # A dict may contain non-string keys with user-defined equality, even in a
    # module namespace. Iteration avoids invoking them during a string lookup.
    for index, (key, value) in enumerate(items):
        if index >= 8192:
            return _UNAVAILABLE  # Do not mistake a bounded search for absence.
        if type(key) is str and key == name:
            return value
    return _MISSING


def _is_direct_class(parts, frame, cls):
    """Read only actual names and plain module dictionaries, never descriptors.

    An optimized frame's locals are Python's own dict/proxy. An exec/class frame
    may use an arbitrary mapping, so only an exact dict is inspected there.
    Builtins, object attributes, metaclass/factory calls and expressions refuse.
    """
    if not parts or frame is None:
        return False
    namespace = frame.f_locals
    if type(namespace) is not dict and not frame.f_code.co_flags & inspect.CO_OPTIMIZED:
        return False
    candidate = _lookup(dict.items(namespace) if type(namespace) is dict else namespace.items(), parts[0])
    if candidate is _MISSING:
        candidate = _lookup(dict.items(frame.f_globals), parts[0])
    for part in parts[1:]:
        if type(candidate) is not types.ModuleType:
            return False
        candidate = _lookup(dict.items(types.ModuleType.__getattribute__(candidate, "__dict__")), part)
    return candidate is cls


def constructor_binding(call):
    if call is None:
        return None
    return {**call, "id": "constructor:" + call["id"], "kind": "constructor"}


class ConstructionEvidence:
    def __init__(self, sources):
        self.sources = sources
        self.instances = {}
        self.seen = 0
        self.limited = False
        self.active = False
        self.argument_bytes = 0

    @contextmanager
    def observe(self):
        # Do not displace another profiler. Missing optional provenance does not
        # prevent the model from being constructed or its forward from running.
        if self.active:
            yield
            return
        if self.limited or sys.getprofile() is not None:
            self.instances.clear()
            yield
            return
        import torch
        self.active = True

        def profile(frame, event, _result):
            if self.limited or event != "call" or frame.f_code.co_name != "__init__":
                return
            try:
                code = frame.f_code
                if not code.co_argcount or not code.co_flags & inspect.CO_OPTIMIZED:
                    return
                module = _lookup(frame.f_locals.items(), code.co_varnames[0])
                # isinstance can invoke an unrelated object's custom __class__
                # property. Profiling must not execute extra model descriptors.
                if not issubclass(type(module), torch.nn.Module):
                    return
                initializer = inspect.getattr_static(type(module), "__init__", None)
                if type(initializer) is not types.FunctionType or initializer.__code__ is not frame.f_code:
                    return  # A super().__init__ call is not another construction.
                self.seen += 1
                if self.seen > MAX_CONSTRUCTIONS:
                    self.limited = True
                    return
                previous = weak_get(self.instances, module)
                caller = frame.f_back
                call = self.sources.expression(caller.f_code, caller.f_lasti) if caller is not None else None
                # Reinitializing one object makes its creation site ambiguous,
                # even if the second initializer has no verified source location.
                verified = previous is None and _is_direct_class(self.sources.constructor_callee(call), caller, type(module))
                arguments, amount = (initializer_arguments.observe_parameters(frame,
                    initializer_arguments.MAX_TOTAL_BYTES - self.argument_bytes) if verified else (None, 0))
                self.argument_bytes += amount
                weak_set(self.instances, module, {
                    "binding": constructor_binding(call) if previous is None else None,
                    "class": weakref.ref(type(module)) if verified else None,
                    "arguments": arguments,
                })
            except Exception:
                # Optional evidence must not replace an exception from the model.
                self.limited = True

        try:
            sys.setprofile(profile)
        except Exception:
            self.active = False
            self.instances.clear()
            yield
            return
        try:
            yield
        finally:
            if sys.getprofile() is profile:
                sys.setprofile(None)
            else:
                # A constructor replaced the observer; earlier entries cannot
                # rule out a later unobserved reinitialization of the same object.
                self.instances.clear()
            self.active = False

    def binding(self, module):
        if self.limited:
            return None  # A missed later reinitialization could invalidate a site.
        record = weak_get(self.instances, module)
        return record["binding"] if record is not None else None

    def clear(self):
        self.instances.clear()
        self.argument_bytes = 0

    def arguments(self, module):
        if not self.class_verified(module):
            return None
        record = weak_get(self.instances, module)
        return record["arguments"] if record is not None else None

    def class_verified(self, module):
        if self.limited:
            return False
        record = weak_get(self.instances, module)
        return bool(record is not None and record["class"] is not None and record["class"]() is type(module))

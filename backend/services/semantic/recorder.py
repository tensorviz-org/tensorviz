"""Canonical observations fed by the existing graph dispatch/module observer.

This adapter owns no dispatch hook and never executes a model on its own. The
small function mode completes metadata after PyTorch's version/autograd updates.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
import sys
import threading
import weakref
from uuid import uuid4

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import _disable_current_modes

from .arguments import ArgumentEncoder
from .contracts import CaptureLimits, CaptureSnapshot, InitializerArguments, Operation, SourceSnapshot
from .diagnostics import forward_diagnostic
from .identity import weak_get, weak_set
from .initializer_arguments import unavailable as unavailable_initializer
from .metrics import ParameterCounter, logical_bytes
from .tensors import TensorMetadata
from . import source_shape_call


def _call_site_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def call_site():
    """The suspended caller instruction distinguishes two calls on one line.

    Only a digest leaves the worker. Capture wrappers are skipped; library
    forward implementations remain valid callers, just like user code.
    """
    services = os.path.dirname(os.path.dirname(__file__)) + os.sep
    frame = sys._getframe(1)
    try:
        for _ in range(256):
            if frame is None:
                break
            file = frame.f_code.co_filename
            if not file.startswith(services) and not file.replace("\\", "/").endswith("/torch/nn/modules/module.py"):
                scope = getattr(frame.f_code, "co_qualname", f"{frame.f_code.co_name}:{frame.f_code.co_firstlineno}")
                return _call_site_digest([file, scope, frame.f_lasti])
            frame = frame.f_back
        return None
    finally:
        del frame


def size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


@dataclass(frozen=True)
class CaptureContext:
    source: SourceSnapshot
    plan_digest: str | None
    attempt_id: str
    limits: CaptureLimits
    input_digest: str | None = None
    state_digest: str | None = None
    preparation_digest: str | None = None
    automatic: dict | None = None
    source_evidence: object | None = None


def automatic_context(preparation, model, args, kwargs, candidate, provenance, source_evidence=None):
    from ..ingest.execution_setup import _check_batch, describe_tree, tree_digest, state_identity
    from .execution_plan import applicability_digest
    nodes = 0
    def builtin(value, depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 16 or nodes > 4096:
            return False
        if type(value) in (torch.Tensor, torch.nn.Parameter, type(None), bool, int, float, str):
            return True
        if type(value) in (list, tuple):
            return len(value) <= 4096 and all(builtin(child, depth + 1) for child in value)
        if type(value) is dict:
            return len(value) <= 4096 and all(type(key) is str and builtin(child, depth + 1) for key, child in value.items())
        return False
    try:
        actual = dict(args=args, kwargs=kwargs)
        if not builtin(actual):
            raise ValueError("unsupported input fingerprint")
        _check_batch(actual)
        input_digest = tree_digest(describe_tree(actual))
    except (ValueError, TypeError, RuntimeError):
        input_digest = None
    return CaptureContext(preparation.source, None, preparation.id, preparation.captureLimits,
        input_digest, state_identity(model)["digest"], applicability_digest(preparation),
        dict(candidateIndex=candidate, candidatesTried=candidate, provenance=provenance), source_evidence)


class SemanticValues(TensorMetadata):
    def __init__(self, recorder):
        self.recorder = recorder
        self.by_id, self.storage_objects, self.object_storage, self.write_epochs = {}, {}, {}, {}
        super().__init__(torch, recorder.data["values"], recorder.limits.values,
                         recorder.omit, recorder.account)

    def snapshot_key(self, oid, version):
        return (oid, version, self.write_epochs.get(oid, 0))

    def observe(self, tensor):
        fact = super().observe(tensor)
        if fact is None:
            return None
        if type(tensor) is FakeTensor:
            self.recorder.fake = True
        self.by_id[fact["id"]] = fact
        oid, sid = fact["objectId"], fact["storageId"]
        old = self.object_storage.get(oid)
        if old == sid:
            return fact
        if old is not None:
            bucket = self.storage_objects.get(old, {})
            bucket.pop(oid, None)
            if not bucket:
                self.storage_objects.pop(old, None)
        self.object_storage[oid] = sid
        def released(reference):
            bucket = self.storage_objects.get(sid, {})
            if bucket.get(oid) is reference:
                del bucket[oid]
                self.object_storage.pop(oid, None)
                if not bucket:
                    self.storage_objects.pop(sid, None)
        self.storage_objects.setdefault(sid, {})[oid] = weakref.ref(tensor, released)
        return fact

    def decorate(self, fact, tensor):
        state = weak_get(self.recorder.state, tensor) or []
        # A FakeTensor parameter keeps nn.Parameter's isinstance identity, not its type.
        parameter = type(tensor) is torch.nn.Parameter or (type(tensor) is FakeTensor and isinstance(tensor, torch.nn.Parameter))
        if parameter and not state:
            self.recorder.omit("unsupported_type")
        previous = self.by_id.get(self.latest.get(fact["objectId"]))
        fact["previousVersion"] = previous["id"] if previous is not None else None
        return dict(**fact, logicalVersion=previous["logicalVersion"] + 1 if previous is not None else 0,
                    stride=list(tensor.stride()), storageOffset=tensor.storage_offset(),
                    logicalBytes=logical_bytes(tensor),
                    state=list(state), producer=None,
                    origin="state" if state else "input" if self.recorder.reading_inputs else "external")


class SemanticRecorder:
    def __init__(self, model, context: CaptureContext):
        self.limits = context.limits
        self.sequence = 0
        self.stack = []
        self.pending = None
        self.writes = []
        self.busy = False
        self.stopped = False
        self.reading_inputs = False
        self.finished = False
        self.thread = threading.get_ident()
        self.omissions = set()
        self.fake = False  # any observed value or registered state was a FakeTensor
        self.diagnostics = {}
        self.instances, self.classes, self.state = {}, {}, {}
        self.parameter_counter = ParameterCounter()
        self.sources = context.source_evidence
        self.bindings = set()
        self.binding_bytes = 0
        self.initializer_bytes = 0
        self.initializers = {}
        self.source_shape_calls, self.source_shape_bytes = {}, 0
        self.source_shape_reads, self.source_shape_read_bytes, self.source_forward_bindings = {}, 0, {}
        self.data = dict(version=1, id=f"capture:{uuid4()}", attemptId=context.attempt_id,
            source=context.source.model_dump(mode="json"), planDigest=context.plan_digest,
            preparationDigest=context.preparation_digest, automatic=context.automatic,
            inputDigest=context.input_digest, stateDigest=context.state_digest,
            backend="runtime", backendVersion="1", torchVersion=str(torch.__version__),
            normalizations=["PyTorch dispatcher operations; Python syntax may be decomposed",
                            "Dispatcher kernels are atomic; arbitrary Python statements are not recorded",
                            "Dictionary subclasses are observed as stored mapping entries; custom attributes and methods are not evaluated",
                            "Logical versions follow schema-declared writes; torch counters remain as observed under dispatch"],
            status="partial", rootInvocationId=None, inputs=dict(args=[], kwargs={}), output=None,
            definitions=[], modules=[], invocations=[], operations=[], values=[], sourceBindings=[],
            diagnostics=[], coverage=dict(scope="executed_path", operationsSeen=0, invocationsSeen=0,
                                         omissions=[], modules=[]), limits=context.limits.model_dump())
        if context.automatic is not None:
            self.data["normalizations"].append("Only the selected input candidate is retained; input search reuses model state across candidates")
        if self.sources is not None:
            self.data["normalizations"].append("Source bindings identify authored expressions, not independently editable dispatcher operations")
        # Reserve bounded final status, omissions and a generic terminal diagnostic.
        self.bytes = size(self.data) + 2048
        self.values = SemanticValues(self)
        self.encoder = ArgumentEncoder(torch, self.tensor, self.omit)
        self.guard(self.inventory, model)

    def omit(self, reason):
        self.omissions.add(reason)
        if reason in ("byte_limit", "recorder_error"):
            self.stopped = True

    def account(self, value, reserve=8):
        if self.stopped:
            return False
        amount = size(value) + reserve
        if self.bytes + amount > self.limits.bytes:
            self.omit("byte_limit")
            return False
        self.bytes += amount
        return True

    def source_binding(self, binding):
        if binding is None:
            return []
        if binding["id"] not in self.bindings:
            amount = size(binding) + 8
            # Optional locations must not consume the whole computational budget.
            if (len(self.bindings) >= 8192 or self.binding_bytes + amount > min(256 * 1024, self.limits.bytes // 8)
                    or self.bytes + amount + 2048 > self.limits.bytes):
                return []
            self.binding_bytes += amount
            self.bytes += amount
            self.bindings.add(binding["id"])
            self.data["sourceBindings"].append(binding)
        return [binding["id"]]

    @contextmanager
    def mute(self):
        prior = self.busy
        self.busy = True
        try:
            with _disable_current_modes(), torch._C.DisableTorchFunction():
                yield
        finally:
            self.busy = prior

    def guard(self, action, *args):
        try:
            with self.mute():
                return action(*args)
        except Exception:
            self.omit("recorder_error")

    def tick(self):
        sequence = self.sequence
        self.sequence += 1
        return sequence

    def inventory(self, model):
        # named_modules(remove_duplicate=False) recurses forever on a cyclic module
        # registration. Traverse paths explicitly while refusing those cycles.
        pending = [(model, "", frozenset())]
        visited = 0
        while pending and not self.stopped:
            module, path, ancestors = pending.pop()
            visited += 1
            if visited > 100000 or len(path) > 4096:
                self.omit("item_limit")
                break
            if id(module) in ancestors:
                self.omit("unsupported_type")
                continue
            self.module(module, path)
            for kind, items in (("parameter", module._parameters), ("buffer", module._buffers)):
                for name, tensor in items.items():
                    if self.stopped:
                        break
                    if tensor is None:
                        continue
                    if type(tensor) is FakeTensor:
                        self.fake = True
                    references = weak_get(self.state, tensor) or []
                    ref = dict(kind=kind, name=f"{path}.{name}" if path else name)
                    if len(references) >= 4096 or len(ref["name"]) > 4096:
                        self.omit("item_limit")
                    elif ref not in references and self.account(ref):
                        weak_set(self.state, tensor, [*references, ref])
            pending.extend((child, f"{path}.{name}" if path else name, ancestors | {id(module)})
                           for name, child in reversed(list(module._modules.items())) if child is not None)

    def module(self, module, path=None):
        found = weak_get(self.instances, module)
        if found is not None:
            if path is not None and path not in found["paths"]:
                if len(found["paths"]) >= 4096:
                    self.omit("item_limit")
                elif self.account(path):
                    found["paths"].append(path)
            return found
        if len(self.data["modules"]) >= 100000:
            self.omit("item_limit")
            return None
        if path is None and (module._parameters or module._buffers):
            self.omit("unsupported_type")  # Runtime state lacks a saved registration identity.
        cls = type(module)
        definition = weak_get(self.classes, cls)
        if definition is None:
            definition = dict(id=f"definition:{len(self.data['definitions'])}",
                qualifiedName=f"{cls.__module__}.{cls.__qualname__}"[:512],
                sourceBindingIds=self.source_binding(self.sources.definition_binding(module)) if self.sources else [])
            if not self.account(definition):
                return None
            self.data["definitions"].append(definition)
            weak_set(self.classes, cls, definition)
        record = dict(id=f"module:{len(self.data['modules'])}", definitionId=definition["id"],
                      paths=[] if path is None else [path],
                      initializerArguments=None,
                      constructorBindingIds=self.source_binding(self.sources.construction.binding(module)) if self.sources else [])
        record["constructorClassVerified"] = bool(record["constructorBindingIds"] and self.sources.construction.class_verified(module))
        if not self.account(record):
            return None
        if record["constructorClassVerified"]:
            arguments = self.sources.construction.arguments(module)
            if arguments is not None:
                amount = size(arguments)
                if self.initializer_bytes + amount > min(262144, self.limits.bytes // 8):
                    arguments = unavailable_initializer("byte_limit")
                    amount = size(arguments)
                if self.initializer_bytes + amount <= min(262144, self.limits.bytes // 8):
                    self.initializer_bytes += amount
                    self.initializers[record["id"]] = arguments
        self.data["modules"].append(record)
        self.source_forward_bindings[record["id"]] = definition["sourceBindingIds"]
        weak_set(self.instances, module, record)
        return record

    def tensor(self, tensor, path):
        if self.stopped:
            return self.encoder.opaque(tensor, "byte_limit" if "byte_limit" in self.omissions else "recorder_error")
        fact = self.values.observe(tensor)
        return (dict(kind="value", id=fact["id"]) if fact is not None
                else self.encoder.opaque(tensor, self.values.reason))

    def enter(self, module, args, kwargs):
        self.flush()
        if self.finished:
            self.omit("unsupported_type")  # A second selected forward is not the first tree.
            self.stack.append(None)
            return
        self.data["coverage"]["invocationsSeen"] += 1
        sequence = self.tick()
        record = None
        if threading.get_ident() != self.thread:
            self.omit("unsupported_type")
        elif not self.stopped and (len(self.data["invocations"]) >= self.limits.invocations or (self.stack and self.stack[-1] is None)):
            self.omit("invocation_limit")
        elif not self.stopped:
            instance = self.module(module)
            self.reading_inputs = not self.stack
            arguments = self.encoder.arguments(args, kwargs)
            self.reading_inputs = False
            if instance is not None:
                candidate = dict(id=f"invocation:{len(self.data['invocations'])}", moduleId=instance["id"],
                    parentId=self.stack[-1]["id"] if self.stack else None, sequence=sequence, endSequence=None,
                    arguments=arguments, output=None, status="interrupted", diagnosticId=None,
                    training=module.training, gradEnabled=torch.is_grad_enabled(),
                    sourceBindingIds=self.source_binding(self.sources.call_binding()) if self.sources else [],
                    **self.parameter_counter.measure(module))
                # Only terminal fields need padding: bounded sequence/diagnostic
                # IDs and an opaque output if its separately charged tree fails.
                if self.account(candidate, 128):
                    record = candidate
                    self.data["invocations"].append(record)
                    if not self.stack:
                        self.data.update(rootInvocationId=record["id"], inputs=arguments)
                        self.account(arguments)  # Root boundaries appear twice on the wire.
        self.stack.append(record)

    def diagnostic(self, error, observed=None):
        facts = dict(forward_diagnostic(error, observed))
        facts.pop("id", None)
        key = json.dumps(facts, sort_keys=True, separators=(",", ":"))
        if key in self.diagnostics:
            return self.diagnostics[key]
        if len(self.data["diagnostics"]) < 4095:
            record = dict(id=f"diagnostic:{len(self.data['diagnostics'])}", **facts)
            if self.account(record):
                self.data["diagnostics"].append(record)
                self.diagnostics[key] = record["id"]
                return record["id"]
        else:
            self.omit("item_limit")
        # The fixed envelope reservation includes one terminal fallback even if
        # capture has stopped. Never leave a failed event with a dangling reference.
        if "fallback" not in self.diagnostics:
            self.data["diagnostics"].append(dict(id="failure", phase="forward", code="execution_failed",
                field=[], names=[], sourceBindingId=None, remedy="inspect_failure"))
            self.diagnostics["fallback"] = "failure"
        return "failure"

    def output(self, output, operation=None):
        start = len(self.data["values"])
        tree = self.encoder.output(output)
        producers = []
        if operation is not None:
            # Only a retained output tree can establish a producer. If an operation
            # returned an existing value, keep its original producer/identity.
            paths = {}
            def walk(item, path=()):
                if item["kind"] == "value":
                    paths.setdefault(item["id"], list(path))
                elif item["kind"] in ("list", "tuple"):
                    for i, child in enumerate(item["items"]):
                        walk(child, (*path, "items", i))
                elif item["kind"] == "dict":
                    for i, entry in enumerate(item["entries"]):
                        walk(entry["value"], (*path, "entries", i, "value"))
            walk(tree)
            for fact in self.data["values"][start:]:
                if fact["id"] in paths:
                    producers.append((fact, dict(operationId=operation["id"], kind="output", path=paths[fact["id"]])))
        # Retain the tree and all its producer paths together. A partial update
        # would leave producer references pointing into an omitted output tree.
        if not self.account(tree, 8 + sum(size(producer) for _, producer in producers)):
            return dict(kind="opaque", reason="recorder_error" if "recorder_error" in self.omissions else "byte_limit", typeName=None)
        for fact, producer in producers:
            fact.update(origin="operation", producer=producer)
        return tree

    def leave(self, output, error=None, observed=None):
        self.flush()
        if not self.stack:
            self.omit("recorder_error")
            return
        record = self.stack.pop()
        if record is not None:
            source_shape_call.end_invocation(self, record["id"])
        sequence = self.tick()
        if record is not None:
            record.update(endSequence=sequence, status="failed" if error is not None else "completed",
                          diagnosticId=self.diagnostic(error, observed) if error is not None else None,
                          output=None if error is not None else self.output(output))
            if not self.stack:
                self.data["output"] = record["output"]
                if record["output"] is not None:
                    self.account(record["output"])
        if not self.stack:
            self.finished = True

    def begin(self, func, args, kwargs):
        self.flush()
        self.writes = []
        if not self.stack or self.finished:
            return None
        self.data["coverage"]["operationsSeen"] += 1
        sequence = self.tick()
        if self.stopped:
            return None
        if len(self.data["operations"]) >= self.limits.operations:
            self.omit("operation_limit")
            return None
        schema = str(func._schema) if hasattr(func, "_schema") else None
        name = str(func)
        if len(name) > 512 or (schema is not None and len(schema) > 4096):
            self.omit("item_limit")
            return None
        record = dict(id=f"operation:{len(self.data['operations'])}", sequence=sequence,
            invocationId=self.stack[-1]["id"] if self.stack[-1] is not None else None,
            target=name, schemaSignature=schema, arguments=self.encoder.arguments(args, kwargs),
            output=None, effects=[], status="interrupted", diagnosticId=None,
            sourceBindingIds=self.source_binding(self.sources.call_binding()) if self.sources else [])
        if not self.account(record, 128):
            return None
        self.data["operations"].append(record)
        self.writes = self.write_targets(func, args, kwargs, record["arguments"])
        return record

    def write_targets(self, func, args, kwargs, arguments):
        """Keep only weak references to exact writable operands and live aliases."""
        targets = []
        def append(kind, path, before, reference):
            if len(targets) >= 16384:
                self.omit("item_limit")
                return
            targets.append((kind, list(path), before, reference))
        def walk(raw, encoded, path):
            if encoded["kind"] == "value":
                append("write", path, encoded["id"], weakref.ref(raw))
            elif encoded["kind"] in ("list", "tuple") and type(raw) in (list, tuple):
                for i, item in enumerate(encoded["items"]):
                    walk(raw[i], item, (*path, "items", i))
        for i, argument in enumerate(getattr(getattr(func, "_schema", None), "arguments", [])):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            if i < len(arguments["args"]):
                walk(args[i], arguments["args"][i], ("args", i))
            elif argument.name in arguments["kwargs"]:
                walk(kwargs[argument.name], arguments["kwargs"][argument.name], ("kwargs", argument.name))
        seen = {self.values.by_id[before]["objectId"] for _, _, before, _ in targets}
        storages = set()
        # Shared-storage views can change without appearing in the operator's return.
        # Indexing by storage avoids scanning every previously observed tensor.
        for _, path, before, _ in list(targets):
            sid = self.values.by_id[before]["storageId"]
            if sid in storages:
                continue
            storages.add(sid)
            for oid, reference in list(self.values.storage_objects.get(sid, {}).items()):
                if len(targets) >= 16384:
                    self.omit("item_limit")
                    break
                if oid not in seen and reference() is not None:
                    seen.add(oid)
                    append("alias_update", path, self.values.latest[oid], reference)
        return targets

    def pending_result(self, record, output):
        if record is None:
            self.writes = []
            return
        for kind, _, _, reference in self.writes:
            tensor = reference()
            if kind == "write" and tensor is not None:
                self.values.written(tensor)
        # Dispatcher kernels do not always advance PyTorch's version counter.
        # Keep that observed counter unchanged; a distinct logical epoch records
        # this declared storage write and its effect on already observed aliases.
        for oid in {self.values.by_id[before]["objectId"] for _, _, before, _ in self.writes}:
            self.values.write_epochs[oid] = self.values.write_epochs.get(oid, 0) + 1
        self.pending = (record, output, self.writes)
        self.writes = []

    def succeeded(self, record, func, args, kwargs, output):
        self.pending_result(record, output)

    def failed(self, record, func=None, error=None, observed=None):
        if record is not None:
            record.update(status="failed", diagnosticId=self.diagnostic(error, observed))
        if func is not None and any(argument.alias_info is not None and argument.alias_info.is_write
                                    for argument in func._schema.arguments):
            self.omit("unsupported_type")  # A failed kernel may have performed part of its write.
        self.pending_result(record, None)

    def flush(self):
        if self.pending is not None:
            record, output, written = self.pending
            self.pending = None
            if record["status"] != "failed":
                record.update(status="completed", output=self.output(output, record))
            for kind, path, before, reference in written:
                tensor = reference()
                fact = self.values.observe(tensor) if tensor is not None and not self.stopped else None
                after = (dict(kind="value", id=fact["id"]) if fact is not None and fact["previousVersion"] == before
                         else self.encoder.opaque(tensor, self.values.reason or "unsupported_type"))
                effect = dict(kind=kind, argumentPath=path, beforeId=before, after=after)
                producer = None
                if after["kind"] == "value" and fact["producer"] is None:
                    producer = dict(operationId=record["id"], kind="effect", path=[len(record["effects"]), "after"])
                if not self.account(effect, 8 + (size(producer) if producer else 0)):
                    continue
                if producer is not None:
                    fact.update(origin="operation", producer=producer)
                record["effects"].append(effect)

    def functions(self):
        recorder = self
        class Functions(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                if recorder.busy or not recorder.stack:
                    return func(*args, **(kwargs or {}))
                escaped = getattr(func, "__name__", "") in ("numpy", "data_ptr", "storage", "untyped_storage")
                start = recorder.data["coverage"]["operationsSeen"]
                if escaped:
                    recorder.omit("unsupported_type")
                source_call = source_shape_call.begin(recorder, func, args, kwargs or {})
                result = None
                try:
                    result = func(*args, **(kwargs or {}))
                    return result
                except BaseException as error:
                    # Argument/autograd checks can fail before dispatcher entry.
                    if recorder.data["coverage"]["operationsSeen"] == start:
                        name = "python:" + getattr(func, "__name__", "torch_operation")
                        record = recorder.guard(recorder.begin, name, args, kwargs or {})
                        recorder.guard(recorder.failed, record, None, error)
                    raise
                finally:
                    recorder.guard(recorder.flush)
                    source_shape_call.finish(recorder, source_call, result)
                    if escaped:
                        recorder.guard(recorder.values.written, args)
        return Functions()

    def snapshot(self):
        self.flush()
        calls = self.data["invocations"]
        by_module = {}
        for call in calls:
            by_module.setdefault(call["moduleId"], []).append(call)
        for module in self.data["modules"]:
            observed = by_module.get(module["id"], [])
            status = ("failed" if any(call["status"] == "failed" for call in observed) else "observed") if observed else (
                "capture_limited" if self.omissions else "not_observed")
            for path in module["paths"]:
                entry = dict(modulePath=path, status=status, invocationIds=[call["id"] for call in observed],
                             reason=sorted(self.omissions)[0] if status == "capture_limited" else None)
                if self.account(entry):
                    self.data["coverage"]["modules"].append(entry)
        root = calls[0] if calls else None
        self.data["coverage"]["omissions"] = sorted(self.omissions)
        self.data["status"] = ("complete" if root and root["status"] == "completed" and not self.omissions
                               else "partial" if calls or self.data["operations"] else "failed")
        if self.fake:
            # Within the fixed terminal reservation. Real captures omit the field.
            self.data["valuesUnavailable"] = "fake_tensor"
        try:
            if size(self.data) > self.limits.bytes:
                self.omit("byte_limit")
                return None  # The immutable source/envelope itself may not fit.
            result = CaptureSnapshot.model_validate(self.data).model_dump(mode="json")
            # Publish optional metadata only after the computational capture has
            # used its budget. Missing constructor observations never truncate a
            # forward that otherwise fits. The default null field is already in
            # the validated payload, so replacement charges only its extra bytes.
            remaining = self.limits.bytes - size(result)
            for module in result["modules"]:
                arguments = self.initializers.get(module["id"])
                if arguments is None:
                    continue
                arguments = InitializerArguments.model_validate(arguments).model_dump(mode="json")
                amount = size(arguments) - 4
                if amount > remaining:
                    arguments = unavailable_initializer("byte_limit")
                    amount = size(arguments) - 4
                if amount <= remaining:
                    module["initializerArguments"] = arguments
                    remaining -= amount
            for operation in result["operations"]:
                call = self.source_shape_calls.get(operation["id"])
                if call is None:
                    continue
                observed = Operation.model_validate({**operation, **call}).model_dump(mode="json")
                amount = size(observed) - size(operation)
                if amount <= remaining:
                    operation.update({key: observed[key] for key in call})
                    remaining -= amount
            return result
        finally:
            self.pending = None
            self.writes = []
            self.instances.clear()
            self.classes.clear()
            self.state.clear()
            self.initializers.clear()
            self.source_shape_calls.clear()
            self.source_shape_reads.clear()
            self.source_forward_bindings.clear()
            self.values.objects.clear()
            self.values.storages.clear()
            self.values.storage_objects.clear()
            self.values.object_storage.clear()

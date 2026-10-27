"""Bounded eager invocation evidence; used only inside an approved execution worker."""
from __future__ import annotations

import json
from itertools import islice
import os
import re
import threading
from ..semantic.identity import weak_identity
from ..semantic.tensors import TensorMetadata


def failure():
    return dict(phase="forward", code="forward_failed", source=None)


def safe_key(key):
    # Only structural identifiers/indices are exposed, never arbitrary user strings.
    if type(key) is int:
        return abs(key) <= 2**53 - 1
    return (type(key) is str and len(key) <= 64
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is not None
            and not re.search(r"secret|password|token|credential|api_?key", key, re.I))


class TraceRecorder:
    def __init__(self, record, torch, sources=None):
        self.record, self.torch = record, torch
        self.sources = sources
        if sources is not None:
            sources.torch_directory = os.path.dirname(torch.__file__) + os.sep
        self.stack = []
        self.tensor_metadata = TensorMetadata(torch, record["tensors"], record["limits"]["max_tensors"],
            lambda reason: self.incomplete("tensor_limit" if reason == "value_limit" else "unsupported"), self.account)
        self.objects, self.storages = self.tensor_metadata.objects, self.tensor_metadata.storages
        self.instances = {}
        self.module_ids = set()
        self.devices = {"cpu"}
        self.producers = {}
        self.counters = {}
        self.bytes = len(json.dumps(record).encode("utf-8"))
        self.function_context = None
        self.pending = None
        self.nodes = 0
        self.disabled = False
        self.reason = None
        self.thread = threading.get_ident()
        self.busy = False
        self.original = self.wrapper = None

    def incomplete(self, reason="unsupported"):
        if self.reason is None or (self.reason == "unsupported" and reason != "unsupported"):
            self.reason = reason
        if reason != "unsupported":
            self.disabled = True

    def account(self, value):
        self.bytes += len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        if self.bytes > self.record["limits"]["max_record_bytes"]:
            self.incomplete("byte_limit")

    def opaque(self, reason="unsupported_type"):
        self.incomplete()
        return dict(kind="opaque", reason=reason)

    def identity(self, value, table, prefix):
        return weak_identity(table, self.counters, value, prefix)

    def tensor(self, value):
        fact = self.tensor_metadata.observe(value)
        if fact is None:
            return self.opaque("item_limit" if self.tensor_metadata.reason == "value_limit" else "unsupported_type")
        self.devices.add(str(value.device))
        return dict(kind="tensor", tensorId=fact["id"])

    def port(self, value, depth=0):
        self.nodes += 1
        if self.disabled or self.nodes > 4096:
            return self.opaque("item_limit")
        if depth >= 8:
            return self.opaque("depth_limit")
        if isinstance(value, self.torch.Tensor):
            return self.tensor(value)
        # Exact built-in types only; arbitrary metadata-like objects stay opaque.
        types = {type(None): "none", bool: "bool", int: "int", float: "float", str: "str",
                 self.torch.dtype: "dtype", self.torch.device: "device",
                 self.torch.layout: "layout", self.torch.memory_format: "memory_format"}
        if type(value) in types:
            return dict(kind="value", type=types[type(value)])
        if type(value) in (list, tuple):
            if len(value) > 256:
                return self.opaque("item_limit")
            return dict(kind="tuple" if type(value) is tuple else "list",
                        items=[self.port(item, depth + 1) for item in value])
        if type(value) is dict:
            if len(value) > 256:
                return self.opaque("item_limit")
            if not all(safe_key(key) for key in value):
                return self.opaque()
            return dict(kind="dict", entries=[dict(key=key, value=self.port(item, depth + 1))
                                               for key, item in value.items()])
        return self.opaque()

    def leaves(self, port, path):
        if port["kind"] == "tensor":
            yield port["tensorId"], path
        elif port["kind"] in ("tuple", "list"):
            for i, item in enumerate(port["items"]):
                yield from self.leaves(item, [*path, i])
        elif port["kind"] == "dict":
            for entry in port["entries"]:
                yield from self.leaves(entry["value"], [*path, entry["key"]])

    def context(self, module):
        devices = set(self.devices)
        # Query enablement without initializing devices or changing any context.
        if self.torch.cuda.is_initialized():
            devices.add(f"cuda:{self.torch.cuda.current_device()}")
        return dict(training=None if module is None else module.training,
                    gradEnabled=self.torch.is_grad_enabled(),
                    inferenceMode=self.torch.is_inference_mode_enabled(),
                    autocast={device: dict(enabled=self.torch.is_autocast_enabled(device.split(":")[0]),
                                           dtype=str(self.torch.get_autocast_dtype(device.split(":")[0])))
                              for device in devices})

    def enter(self, name, args, kwargs, module=None, opaque=False):
        self.flush()
        if self.disabled:
            return None
        if len(self.record["invocations"]) >= self.record["limits"]["max_events"]:
            self.incomplete("event_limit")
            return None
        if module is not None:
            symbol = f"{type(module).__module__}:{type(module).__qualname__}"
            if "<" in symbol:
                self.incomplete()
                if not self.stack:
                    # The contract requires a symbol-identifiable module root.
                    self.disabled = True
                    return None
                module, opaque, name = None, True, "unsupported_module"
        self.nodes = 0
        if len(args) > 64 or len(kwargs) > 64:
            self.incomplete()
        inputs = [self.port(value) for value in args[:64]]
        keywords = {}
        for key, value in islice(kwargs.items(), 64):
            if type(key) is str and safe_key(key):
                keywords[key] = self.port(value)
            else:
                self.incomplete()
        mid = None
        source = None
        if module is not None:
            source = self.sources.module(module) if self.sources is not None else None
            mid = self.identity(module, self.instances, "module")
            if mid not in self.module_ids:
                self.record["modules"].append(dict(id=mid, classSymbol=symbol, source=source))
                self.module_ids.add(mid)
                self.account(self.record["modules"][-1])
        elif self.sources is not None:
            source = self.sources.callsite()
        context = (self.function_context if module is None and self.function_context is not None
                   else self.context(module))
        call = dict(id=f"call{len(self.record['invocations'])}",
                    parentId=self.stack[-1]["id"] if self.stack else None, moduleId=mid,
                    kind="module" if module is not None else "opaque" if opaque else "function",
                    name=name, source=source, context=context, inputs=inputs, kwargs=keywords,
                    outcome="failed", output=None, failure=failure())
        self.record["invocations"].append(call)
        self.account(call)
        if not self.stack:
            self.record["capture"]["rootInvocationId"] = call["id"]
            self.record["context"] = dict(**context, torchVersion=str(self.torch.__version__), devices=[])
        for slot, ports in (("args", enumerate(inputs)), ("kwargs", keywords.items())):
            for key, port in ports:
                for tid, path in self.leaves(port, [key]):
                    if self.disabled:
                        break
                    if len(self.record["edges"]) >= 100000:
                        self.incomplete("event_limit")
                        break
                    self.record["edges"].append(dict(tensorId=tid, producer=self.producers.get(tid),
                        consumer=dict(invocationId=call["id"], slot=slot, path=path)))
                    self.account(self.record["edges"][-1])
        if opaque:
            self.incomplete()
        return call

    def succeeded(self, call, output):
        call.update(outcome="unsupported" if call["kind"] == "opaque" else "succeeded",
                    output=dict(kind="opaque", reason="unsupported_type"), failure=None)
        self.nodes = 0
        call["output"] = self.port(output)
        self.account(call["output"])
        for tid, path in self.leaves(call["output"], []):
            # A containing module's return must not replace an actual operation producer.
            self.producers.setdefault(tid, dict(invocationId=call["id"], slot="output", path=path))

    def failed(self, call, error):
        if call is not None and self.sources is not None:
            call["failure"]["source"] = self.sources.failure(error) or (
                call["source"] if call["kind"] != "module" else None)
            self.account(call["failure"])

    def guard(self, action, *args):
        self.busy = True
        try:
            return action(*args)
        except Exception:
            self.incomplete("recorder_error")
        finally:
            self.busy = False

    def flush(self):
        if self.pending is not None:
            call, output = self.pending
            self.pending = None
            self.succeeded(call, output)

    def module_call(self, module, args, kwargs, invoke):
        if threading.get_ident() != self.thread:
            self.incomplete()
            return invoke()
        call = self.guard(self.enter, "forward", args, kwargs, module)
        if call is not None:
            self.stack.append(call)
        try:
            output = invoke()
        except BaseException as error:
            self.guard(self.flush)
            self.guard(self.failed, call, error)
            raise
        else:
            self.guard(self.flush)
            if call is not None:
                self.guard(self.succeeded, call, output)
            return output
        finally:
            if call is not None:
                self.stack.pop()

    def install(self, skip_module):
        self.original = self.torch.nn.Module._call_impl

        def observe(module, *args, **kwargs):
            nonlocal skip_module
            if module is skip_module:
                skip_module = None
                return self.original(module, *args, **kwargs)
            return self.module_call(module, args, kwargs, lambda: self.original(module, *args, **kwargs))

        self.wrapper = observe
        self.torch.nn.Module._call_impl = observe

    def written(self, value):
        self.tensor_metadata.written(value)

    def modes(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.overrides import TorchFunctionMode
        recorder = self

        class Dispatch(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if recorder.busy:
                    return func(*args, **kwargs)
                opaque = func.namespace != "aten"
                call = recorder.guard(recorder.enter, str(func) if not opaque else "custom_operation",
                                      args, kwargs, None, opaque)
                # Entry is recorded before dispatch, including failed attempted inputs.
                try:
                    output = func(*args, **kwargs)
                except BaseException as error:
                    recorder.guard(recorder.failed, call, error)
                    raise
                if call is not None:
                    for i, argument in enumerate(func._schema.arguments):
                        if argument.alias_info is not None and argument.alias_info.is_write:
                            value = args[i] if i < len(args) else kwargs.get(argument.name)
                            recorder.guard(recorder.written, value)
                    # Autograd metadata/version increments occur ABOVE dispatch. Complete
                    # the output at the Python return (or next dispatch), never guess +1.
                    recorder.pending = (call, output)
                return output

        class Functions(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                if recorder.busy:
                    return func(*args, **(kwargs or {}))
                prior = recorder.function_context
                recorder.function_context = recorder.guard(recorder.context, None)
                start = len(recorder.record["invocations"])
                boundary = None
                if getattr(func, "__name__", "") in ("numpy", "data_ptr", "storage", "untyped_storage"):
                    boundary = recorder.guard(recorder.enter, "external_tensor_access", args, kwargs or {}, None, True)
                try:
                    output = func(*args, **(kwargs or {}))
                    if boundary is not None:
                        recorder.guard(recorder.flush)
                        recorder.guard(recorder.succeeded, boundary, output)
                        # The escaped buffer may change without a PyTorch version bump.
                        recorder.guard(recorder.written, args)
                    return output
                except BaseException as error:
                    recorder.guard(recorder.flush)
                    # Python argument/autograd checks can fail before reaching dispatch.
                    if not any(call["outcome"] == "failed" for call in recorder.record["invocations"][start:]):
                        name = getattr(func, "__name__", "torch_operation")
                        call = recorder.guard(recorder.enter, name, args, kwargs or {})
                        recorder.guard(recorder.failed, call, error)
                    if boundary is not None:
                        recorder.guard(recorder.failed, boundary, error)
                    raise
                finally:
                    recorder.guard(recorder.flush)
                    recorder.function_context = prior

        return Functions(), Dispatch()

    def run(self, model, args, kwargs, invoke=None):
        self.install(model if invoke is None else None)
        try:
            modes = self.guard(self.modes)
            if modes is None:
                return self.module_call(model, args, kwargs,
                    invoke if invoke is not None else lambda: model(*args, **kwargs))
            functions, dispatch = modes
            with functions, dispatch:
                # Script selection already entered the original wrapper; direct mode
                # enters here too so both modes have exactly one selected root.
                return self.module_call(model, args, kwargs,
                    invoke if invoke is not None else lambda: model(*args, **kwargs))
        finally:
            if self.torch.nn.Module._call_impl is self.wrapper:
                self.torch.nn.Module._call_impl = self.original
            self.guard(self.flush)
            self.pending = None
            self.wrapper = self.original = None
            for table in (self.objects, self.storages, self.instances):
                table.clear()
            if self.sources is not None:
                self.sources.clear()
            root = self.record["capture"]["rootInvocationId"]
            self.record["capture"] = (dict(state="incomplete", rootInvocationId=root, reason=self.reason)
                                      if self.reason else dict(state="complete", rootInvocationId=root))
            if self.record["context"] is not None:
                self.record["context"]["devices"] = sorted({t["device"] for t in self.record["tensors"]})

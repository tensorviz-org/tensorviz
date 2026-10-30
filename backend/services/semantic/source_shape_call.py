"""Optional direct Python-to-ATen reshape and flatten argument observations.

Only ordinary tensor calls, literal integer values and exact source locations are
retained. No source evaluation, descriptor access or tensor keepalive is added.
"""
import json
import inspect
import types
import weakref

import torch

from .identity import weak_get
from .contracts import SourceFlattenCall

MAX_CALLS = 1024
MAX_BYTES = 256 * 1024
MAX_READ_CALLS = 1024
MAX_READ_BYTES = 256 * 1024
_RESHAPE = torch.reshape
_FLATTEN = torch.flatten
_TENSOR, _PARAMETER = torch.Tensor, torch.nn.Parameter
_SIZE = torch.Size
_SHAPE = inspect.getattr_static(_TENSOR, "shape")
_TENSOR_LOOKUP = inspect.getattr_static(_TENSOR, "__getattribute__")
_PARAMETER_LOOKUP = inspect.getattr_static(_PARAMETER, "__getattribute__")
_METHODS = ((_TENSOR.reshape, "reshape"), (_TENSOR.view, "view"), (_TENSOR.flatten, "flatten"))
_MODULE_CALL = torch.nn.Module._call_impl.__code__
_MODULE_CALL_CODES = (_MODULE_CALL, *[code for code in _MODULE_CALL.co_consts
    if type(code) is types.CodeType and code.co_name == "inner"])


def tensor_read(recorder, value, axis):
    """Read existing metadata only; never call shape or register a new value."""
    if type(value) is not _TENSOR and type(value) is not _PARAMETER:
        return None
    lookup = _TENSOR_LOOKUP if type(value) is _TENSOR else _PARAMETER_LOOKUP
    if (inspect.getattr_static(type(value), "shape", None) is not _SHAPE
            or inspect.getattr_static(type(value), "__getattribute__", None) is not lookup):
        return None
    identity = weak_get(recorder.values.objects, value)
    value_id = recorder.values.latest.get(identity)
    fact = recorder.values.by_id.get(value_id)
    if fact is None or fact["shape"] is None:
        return None
    index = axis if axis >= 0 else len(fact["shape"]) + axis
    if not 0 <= index < len(fact["shape"]):
        return None
    return dict(valueId=value_id, axis=axis, observed=fact["shape"][index])


def take_reads(recorder, key):
    entry = recorder.source_shape_reads.pop(key, None)
    if entry is None:
        return {}
    recorder.source_shape_read_bytes -= entry["bytes"]
    return entry["reads"]


def end_invocation(recorder, invocation_id):
    for key in tuple(recorder.source_shape_reads):
        if key[0] == invocation_id:
            take_reads(recorder, key)


def read_key(recorder, frame_id, call_id):
    invocation = recorder.stack[-1] if recorder.stack else None
    return (invocation["id"], frame_id, call_id) if invocation else None


def flatten_call(binding, method, args, kwargs):
    if (len(args) > 3 or set(kwargs) - {"start_dim", "end_dim", *([] if method else ["input"])}
            or method and not args or not method and args and "input" in kwargs):
        return None
    call = dict(callable="torch.Tensor.flatten" if method else "torch.flatten", bindingId=binding["id"], arguments=[])
    for index, (name, field, default) in enumerate((("start_dim", "startDim", 0), ("end_dim", "endDim", -1))):
        if len(args) > index + 1:
            if name in kwargs:
                return None
            value, path = args[index + 1], ["args", index if method else index + 1]
        elif name in kwargs:
            value, path = kwargs[name], ["kwargs", name]
        else:
            value, path = default, None
        if type(value) is not int or not -32 <= value <= 31:
            return None
        call[field] = value
        if path is not None:
            call["arguments"].append(dict(parameter=name, sourcePath=path, value=value))
    return call


def begin(recorder, func, args, kwargs):
    if recorder.sources is None or len(recorder.source_shape_calls) >= MAX_CALLS:
        return None
    method = next((name for target, name in _METHODS if func is target), None)
    getter = type(func) is types.MethodWrapperType and func.__self__ is _SHAPE and func.__name__ == "__get__"
    if not getter and func is not _RESHAPE and func is not _FLATTEN and method is None:
        return None
    try:
        with recorder.mute():
            if getter:
                if type(args) is not tuple or len(args) != 1 or kwargs:
                    return None
                source = recorder.sources.shape_read_site(_MODULE_CALL_CODES)
                invocation = recorder.stack[-1] if recorder.stack else None
                if source is None or invocation is None:
                    return None
                site, frame_id = source
                if site["definitionId"] not in recorder.source_forward_bindings.get(invocation["moduleId"], []):
                    return None
                read = tensor_read(recorder, args[0], site["axis"])
                return None if read is None else dict(kind="tensor_read", read=read, path=tuple(site["path"]),
                    key=read_key(recorder, frame_id, site["callId"]))
            # A project replacement installed before this module was loaded must
            # not inherit the standard-call observation grammar.
            if type(func) is not types.BuiltinFunctionType and type(func) is not types.MethodDescriptorType:
                return None
            if type(args) is not tuple or type(kwargs) is not dict or any(type(key) is not str for key in kwargs):
                return None
            receiver = args[0] if args else kwargs.get("input")
            if type(receiver) is not _TENSOR and type(receiver) is not _PARAMETER:
                return None
            source = recorder.sources.direct_shape_call(func, receiver, method)
            if source is None:
                return None
            binding, frame_id = source
            reads = take_reads(recorder, read_key(recorder, frame_id, binding["id"]))
            if func is _FLATTEN or method == "flatten":
                call = flatten_call(binding, method, args, kwargs)
                return None if call is None else dict(start=len(recorder.data["operations"]),
                    seen=recorder.data["coverage"]["operationsSeen"], receiver=weakref.ref(receiver),
                    field="sourceFlattenCall", call=call)
            if method:
                if not args or kwargs or len(args) < 2:
                    return None
                arguments = [(["args", index], value) for index, value in enumerate(args[1:])]
            else:
                if len(args) > 2 or set(kwargs) - {"input", "shape"} or (args and "input" in kwargs) or (len(args) > 1 and "shape" in kwargs):
                    return None
                arguments = [(["args", 1], args[1])] if len(args) > 1 else [(["kwargs", "shape"], kwargs.get("shape"))]
            dimensions = []
            for path, value in arguments:
                if type(value) is list or type(value) is tuple:
                    if len(arguments) != 1 or len(value) > 32:
                        return None
                    leaves = [(path + ["items", index], item) for index, item in enumerate(value)]
                else:
                    leaves = [(path, value)]
                for source_path, item in leaves:
                    if type(item) is not int or abs(item) > 9007199254740991 or len(dimensions) >= 32:
                        return None
                    dimensions.append(dict(sourcePath=source_path, dimension=len(dimensions), value=dict(kind="int", value=str(item))))
            if not dimensions:
                return None
            return dict(start=len(recorder.data["operations"]), seen=recorder.data["coverage"]["operationsSeen"],
                receiver=weakref.ref(receiver), reads=reads, field="sourceShapeCall", call=dict(callable="torch.Tensor." + method if method else "torch.reshape",
                bindingId=binding["id"], dimensions=dimensions))
    except Exception:
        return None  # Optional provenance never changes the model's outcome.


def finish(recorder, observation, result=None):
    if observation is None:
        return
    try:
        with recorder.mute():
            if observation.get("kind") == "tensor_read":
                read, key, path = observation["read"], observation["key"], observation["path"]
                axis = read["axis"]
                if (key is None or type(result) is not _SIZE or not -len(result) <= axis < len(result)
                        or type(result[axis]) is not int or result[axis] != read["observed"]):
                    return
                amount = len(json.dumps([key, path, read], separators=(",", ":")).encode())
                if (recorder.source_shape_read_bytes + amount > min(MAX_READ_BYTES, recorder.limits.bytes // 8)
                        or key not in recorder.source_shape_reads and len(recorder.source_shape_reads) >= MAX_READ_CALLS):
                    recorder.source_shape_reads.clear()
                    recorder.source_shape_read_bytes = 0
                    return
                entry = recorder.source_shape_reads.setdefault(key, dict(reads={}, bytes=0))
                if len(entry["reads"]) >= 32:
                    take_reads(recorder, key)
                    return
                # A repeated instruction is outside the single-pass grammar.
                entry["reads"][path] = None if path in entry["reads"] else read
                entry["bytes"] += amount
                recorder.source_shape_read_bytes += amount
                return
            if (len(recorder.data["operations"]) != observation["start"] + 1
                    or recorder.data["coverage"]["operationsSeen"] != observation["seen"] + 1):
                return
            operation = recorder.data["operations"][observation["start"]]
            call = observation["call"]
            arguments = operation["arguments"]["args"]
            receiver = observation["receiver"]()
            if (receiver is None or operation["target"] != "aten.view.default" or operation["status"] != "completed"
                    or operation["effects"] or operation["sourceBindingIds"] != [call["bindingId"]]
                    or len(arguments) != 2 or operation["arguments"]["kwargs"] or arguments[0]["kind"] != "value"
                    or arguments[1]["kind"] not in ("list", "tuple") or operation["output"]["kind"] != "value"
                    or recorder.values.by_id[arguments[0]["id"]]["objectId"] != weak_get(recorder.values.objects, receiver)):
                return
            if observation["field"] == "sourceFlattenCall":
                shape = recorder.values.by_id[arguments[0]["id"]]["shape"]
                dimensions = SourceFlattenCall.model_validate(call).output_dimensions(shape)
                if (dimensions is None or recorder.values.by_id[operation["output"]["id"]]["shape"] != dimensions
                        or arguments[1]["items"] != [dict(kind="int", value=str(d)) for d in dimensions]):
                    return
            else:
                if arguments[1]["items"] != [item["value"] for item in call["dimensions"]]:
                    return
                for dimension in call["dimensions"]:
                    read = observation["reads"].get(tuple(dimension["sourcePath"]))
                    if read is None or read["observed"] != int(dimension["value"]["value"]):
                        continue
                    dimension["tensorDimension"] = dict(valueId=read["valueId"], axis=read["axis"])
            amount = len(json.dumps(call, separators=(",", ":")).encode())
            if recorder.source_shape_bytes + amount > min(MAX_BYTES, recorder.limits.bytes // 8):
                return
            recorder.source_shape_calls[operation["id"]] = {observation["field"]: call}
            recorder.source_shape_bytes += amount
    except Exception:
        return  # Unknown correspondence is represented by absence, never a guess.

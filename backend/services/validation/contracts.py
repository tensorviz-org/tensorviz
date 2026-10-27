"""Python-owned v1 data contracts. No imports, resolution or execution of user code.

See docs/pytorch-validation-contracts.md for binding, identity and transport rules.
The exported JSON schema/TypeScript are structural contracts; the validators here
also check relationships (tensor values, references and lifecycle consistency).
"""
from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

VERSION = 1
MAX_SAFE_INTEGER = 2**53 - 1
CONFIG_MAX_BYTES = 262144
RECORD_MAX_BYTES = 4194304
WORKER_FRAME_MAX_BYTES = RECORD_MAX_BYTES + 4096
WORKER_HELLO_MAX_BYTES = 1024
WORKER_MAX_RECORDS = 8
BOOTSTRAP_MAX_BYTES = 16777216
MAX_ARGUMENT_DEPTH = 8
MAX_ARGUMENT_NODES = 4096

SafeInt = Annotated[StrictInt, Field(ge=-MAX_SAFE_INTEGER, le=MAX_SAFE_INTEGER)]
Nat = Annotated[StrictInt, Field(ge=0, le=MAX_SAFE_INTEGER)]
Positive = Annotated[StrictInt, Field(ge=1, le=MAX_SAFE_INTEGER)]


# Generator numbers are deliberately bounded; tagged Python floats use FiniteFloat.
FiniteFloat = Annotated[StrictFloat, Field(allow_inf_nan=False)]
Number = Union[SafeInt, Annotated[FiniteFloat, Field(ge=-MAX_SAFE_INTEGER, le=MAX_SAFE_INTEGER)]]
Text = Annotated[StrictStr, Field(max_length=4096)]
Name = Annotated[StrictStr, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}(?![\s\S])")]
Id = Annotated[StrictStr, Field(min_length=1, max_length=128)]
Digest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}(?![\s\S])")]
# Portable project-relative paths, normalized without '.'/'..' or shell expansion.
Path = Annotated[StrictStr, Field(min_length=1, max_length=1024, pattern=r"^(?!/)(?!.*//)(?!.*/$)(?!.*(?:^|/)\.{1,2}(?:/|$))[^\\:\x00-\x1f]+(?![\s\S])")]
Symbol = Annotated[StrictStr, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?![\s\S])", max_length=512)]
Device = Annotated[StrictStr, Field(max_length=64, pattern=r"^(?:cpu|mps|cuda:(?:0|[1-9][0-9]*))(?![\s\S])")]
Dtype = Literal["bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64"]
Scalar = Union[StrictBool, SafeInt, Text]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, regex_engine="python-re", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def exact_version(cls, value):
        if ("version" in cls.model_fields and cls.model_fields["version"].annotation == Literal[1]
                and isinstance(value, dict) and "version" in value and type(value["version"]) is not int):
            raise ValueError("version must be an integer, not a boolean or float")
        return value


class FloatValue(Contract):
    # JSON through JavaScript erases 1.0 vs 1. Materialize this tag with float(value).
    kind: Literal["float"]
    value: FiniteFloat


class NoneValue(Contract):
    kind: Literal["none"]


class SequenceValue(Contract):
    kind: Literal["list", "tuple"]
    items: Annotated[list["Argument"], Field(max_length=64)]


class DictValue(Contract):
    kind: Literal["dict"]
    items: Annotated[dict[Annotated[StrictStr, Field(max_length=128)], "Argument"], Field(max_length=64)]


class Normal(Contract):
    kind: Literal["normal"]
    mean: Number
    std: Annotated[Number, Field(gt=0)]
    seed: Nat


class Uniform(Contract):
    kind: Literal["uniform"]
    low: Number
    high: Number
    seed: Nat

    @model_validator(mode="after")
    def ordered(self):
        if self.low >= self.high:
            raise ValueError("low must be less than high (exclusive)")
        return self


class RandInt(Contract):
    kind: Literal["randint"]
    low: SafeInt
    high: SafeInt
    seed: Nat


class Full(Contract):
    kind: Literal["full"]
    value: Union[StrictBool, Number]


class Values(Contract):
    kind: Literal["values"]
    values: Annotated[list[Union[StrictBool, Number]], Field(max_length=4096)]


Generator = Annotated[Union[Normal, Uniform, RandInt, Full, Values], Field(discriminator="kind")]
_INTEGER_BOUNDS = {"uint8": (0, 255), "int8": (-128, 127), "int16": (-32768, 32767),
                   "int32": (-2**31, 2**31 - 1), "int64": (-MAX_SAFE_INTEGER, MAX_SAFE_INTEGER)}


class TensorValue(Contract):
    kind: Literal["tensor"]
    shape: Annotated[list[Annotated[StrictInt, Field(ge=0, le=2**31 - 1)]], Field(max_length=32)]
    dtype: Dtype
    requires_grad: StrictBool
    generator: Generator

    @model_validator(mode="after")
    def compatible_values(self):
        integral = self.dtype in _INTEGER_BOUNDS
        floating = self.dtype.startswith("float") or self.dtype == "bfloat16"
        gen = self.generator
        if self.requires_grad and not floating:
            raise ValueError("requires_grad needs a floating dtype")
        if isinstance(gen, (Normal, Uniform)) and not floating:
            raise ValueError("normal/uniform require a floating dtype")
        if isinstance(gen, RandInt):
            if not integral or not gen.low < gen.high:
                raise ValueError("randint requires integer dtype and low < high")
            lo, hi = _INTEGER_BOUNDS[self.dtype]
            if gen.low < lo or gen.high - 1 > hi:
                raise ValueError("randint bounds exceed dtype range")
        if isinstance(gen, Values) and len(gen.values) != math.prod(self.shape):
            raise ValueError("flat values count must equal the product of shape (scalar needs one)")
        values = gen.values if isinstance(gen, Values) else [gen.value] if isinstance(gen, Full) else []
        for value in values:
            if self.dtype == "bool" and type(value) is not bool:
                raise ValueError("bool tensors require boolean literal values")
            if integral:
                lo, hi = _INTEGER_BOUNDS[self.dtype]
                if type(value) is not int or not lo <= value <= hi:
                    raise ValueError("integer literal outside dtype range")
            if floating and type(value) is bool:
                raise ValueError("floating tensors require numeric literal values")
        return self


Argument = Union[Scalar, Annotated[Union[FloatValue, NoneValue, SequenceValue, DictValue, TensorValue], Field(discriminator="kind")]]


class Binding(Contract):
    args: Annotated[list[Argument], Field(max_length=64)]
    kwargs: Annotated[dict[Name, Argument], Field(max_length=64)]

    @model_validator(mode="before")
    @classmethod
    def bounded_tree(cls, value):
        nodes = 0

        def visit(item, depth):
            nonlocal nodes
            nodes += 1
            if nodes > MAX_ARGUMENT_NODES or depth > MAX_ARGUMENT_DEPTH:
                raise ValueError("argument tree exceeds depth/node limit")
            if isinstance(item, dict) and item.get("kind") in ("list", "tuple", "dict"):
                children = item.get("items", [])
                for child in children.values() if isinstance(children, dict) else children if isinstance(children, list) else []:
                    visit(child, depth + 1)
        if isinstance(value, dict):
            args, kwargs = value.get("args", []), value.get("kwargs", {})
            for item in args if isinstance(args, list) else []:
                visit(item, 0)
            for item in kwargs.values() if isinstance(kwargs, dict) else []:
                visit(item, 0)
        return value


class Limits(Contract):
    # Only resource defaults, never semantic input defaults. Resolved specs carry all fields.
    timeout_ms: Annotated[StrictInt, Field(ge=100, le=86400000)] = 60000
    max_events: Annotated[StrictInt, Field(ge=1, le=100000)] = 10000
    max_tensors: Annotated[StrictInt, Field(ge=1, le=100000)] = 10000
    max_record_bytes: Annotated[StrictInt, Field(ge=1024, le=RECORD_MAX_BYTES)] = RECORD_MAX_BYTES
    # Graph capture is independent of the bounded validation-record transport.
    max_capture_bytes: Annotated[StrictInt, Field(ge=1024, le=16777216)] | None = None


class CaptureSelector(Contract):
    model: Symbol
    call: Positive


class ProfileBase(Contract):
    cwd: Union[Literal["."], Path] = "."
    tracked_files: Annotated[list[Path], Field(max_length=256)] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def unique_paths(self):
        if len(self.tracked_files) != len(set(self.tracked_files)):
            raise ValueError("tracked_files contains duplicates")
        return self


class ScriptProfile(ProfileBase):
    kind: Literal["script"]
    script: Path
    args: Annotated[list[Text], Field(max_length=256)]
    capture: CaptureSelector


class ModelFactory(Contract):
    file: Path
    callable: Name
    kwargs: Annotated[dict[Name, JsonValue], Field(max_length=64)] = Field(default_factory=dict)

    @model_validator(mode="after")
    def bounded_callable(self):
        from .setup import callable_spec
        callable_spec(self.model_dump())
        return self


class InputFactory(Contract):
    factory: ModelFactory


class ModelProfile(ProfileBase):
    model_config = ConfigDict(json_schema_extra={"oneOf": [
        {"required": ["init"], "properties": {"init": {"not": {"type": "null"}}, "factory": {"type": "null"}}},
        {"required": ["factory"], "properties": {"factory": {"not": {"type": "null"}}, "init": {"type": "null"}}},
    ]})
    kind: Literal["model"]
    framework: Literal["jax"] | None = None
    model: Symbol
    mode: Literal["train", "eval"]
    device: Device
    grad_enabled: StrictBool
    deterministic: StrictBool = True
    seed: Nat | None = None
    init: Binding | None = None
    factory: ModelFactory | None = None
    inputs: Union[Binding, InputFactory]

    @model_validator(mode="after")
    def one_constructor(self):
        if (self.init is None) == (self.factory is None):
            raise ValueError("provide exactly one of init or factory")
        return self



Profile = Annotated[Union[ScriptProfile, ModelProfile], Field(discriminator="kind")]


class ValidationConfig(Contract):
    version: Literal[1]
    profiles: Annotated[dict[Name, Profile], Field(min_length=1, max_length=64)]


class Requirement(Contract):
    profile: Name | None
    field: Annotated[list[Union[Annotated[StrictStr, Field(max_length=128)], Nat]], Field(min_length=1, max_length=32)]
    phase: Literal["static", "runtime"]
    code: Literal["missing", "invalid", "invalid_source", "unknown_key", "ambiguous", "unsupported", "unsaved", "missing_file", "environment", "signature", "removed_format"]
    reason: Annotated[StrictStr, Field(min_length=1, max_length=512)]
    remedy: Annotated[StrictStr, Field(min_length=1, max_length=512)]


class FileIdentity(Contract):
    path: Path
    digest: Digest
    documentVersion: Nat | None
    role: Literal["config", "source", "native"]


class RunIdentity(Contract):
    workspaceSessionId: Id
    workspaceId: Id
    runId: Id
    generation: Nat
    profile: Name
    configDigest: Digest
    profileDigest: Digest
    sourceDigest: Digest
    environmentDigest: Digest


class InterpreterIdentity(Contract):
    executable: Annotated[StrictStr, Field(min_length=1, max_length=4096)]
    version: Id
    environmentDigest: Digest


class RunSpec(Contract):
    version: Literal[1]
    identity: RunIdentity
    profile: Profile
    files: Annotated[list[FileIdentity], Field(min_length=1, max_length=100000)]
    dependencyScope: Literal["complete", "partial"]
    interpreter: InterpreterIdentity

    @model_validator(mode="after")
    def consistent_identity(self):
        paths = [file.path for file in self.files]
        configs = [file for file in self.files if file.role == "config"]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate file identity")
        if len(configs) != 1 or configs[0].path != "tensorviz.toml" or configs[0].digest != self.identity.configDigest:
            raise ValueError("one saved tensorviz.toml identity must match configDigest")
        if self.interpreter.environmentDigest != self.identity.environmentDigest:
            raise ValueError("interpreter identity mismatch")
        required = set(self.profile.tracked_files)
        if isinstance(self.profile, ScriptProfile):
            required.add(self.profile.script)
        elif self.profile.factory is not None:
            required.add(self.profile.factory.file)
            if not any(file.path == self.profile.factory.file and file.role == "source" for file in self.files):
                raise ValueError("factory requires a saved source identity")
        if isinstance(self.profile, ModelProfile) and isinstance(self.profile.inputs, InputFactory):
            required.add(self.profile.inputs.factory.file)
            if not any(file.path == self.profile.inputs.factory.file and file.role == "source" for file in self.files):
                raise ValueError("input factory requires a saved source identity")
        if not required.issubset(paths):
            raise ValueError("script/tracked file identity missing")
        return self


class Resolved(Contract):
    status: Literal["resolved"]
    spec: RunSpec


class NeedsSetup(Contract):
    status: Literal["needs_setup"]
    requirements: Annotated[list[Requirement], Field(min_length=1, max_length=256)]


Resolution = Annotated[Union[Resolved, NeedsSetup], Field(discriminator="status")]


class SourceLocation(Contract):
    file: Path
    line: Positive
    column: Nat


class ModuleInstance(Contract):
    id: Id
    classSymbol: Symbol
    source: SourceLocation | None


class TensorFact(Contract):
    id: Id  # one observed tensor object AND version; never a Python address
    objectId: Id
    storageId: Id | None
    version: Nat | None  # inference/opaque tensors may not expose a version
    shape: Annotated[list[Nat], Field(max_length=32)] | None
    dtype: Id
    device: Id
    requiresGrad: StrictBool
    aliasOf: Id | None  # observed view/alias, not shape equality
    previousVersion: Id | None


class TensorPort(Contract):
    kind: Literal["tensor"]
    tensorId: Id


class ValuePort(Contract):
    kind: Literal["value"]
    # Closed type-only ports: never serialize argument values or object repr/state.
    type: Literal["none", "bool", "int", "float", "str", "dtype", "device", "layout", "memory_format"]


class OpaquePort(Contract):
    kind: Literal["opaque"]
    reason: Literal["unsupported_type", "depth_limit", "item_limit"]


class SequencePort(Contract):
    kind: Literal["list", "tuple"]
    items: Annotated[list["Port"], Field(max_length=256)]


class DictPort(Contract):
    kind: Literal["dict"]
    entries: Annotated[list["DictEntry"], Field(max_length=256)]


class DictEntry(Contract):
    key: Union[Annotated[StrictStr, Field(max_length=128)], SafeInt]  # string/int keys only; opaque for other key types
    value: "Port"


Port = Annotated[Union[TensorPort, ValuePort, OpaquePort, SequencePort, DictPort], Field(discriminator="kind")]


class Failure(Contract):
    phase: Literal["setup", "construction", "forward", "script", "recorder", "transport", "resource"]
    code: Id
    # Sanitized diagnostic code and source only. No repr(exception), source, argv, secrets.
    source: SourceLocation | None


class AutocastContext(Contract):
    enabled: StrictBool
    dtype: Id | None

    @model_validator(mode="after")
    def enabled_dtype(self):
        if self.enabled and self.dtype is None:
            raise ValueError("enabled autocast requires its observed dtype")
        return self


class CallContext(Contract):
    training: StrictBool | None
    gradEnabled: StrictBool
    inferenceMode: StrictBool
    autocast: Annotated[dict[Device, AutocastContext], Field(max_length=16)]


class ObservedContext(CallContext):
    torchVersion: Id
    training: StrictBool
    devices: Annotated[list[Device], Field(max_length=16)]


class Invocation(Contract):
    id: Id
    parentId: Id | None
    moduleId: Id | None
    kind: Literal["module", "function", "opaque"]
    name: Annotated[StrictStr, Field(min_length=1, max_length=256)]
    source: SourceLocation | None
    context: CallContext
    inputs: Annotated[list[Port], Field(max_length=64)]
    kwargs: Annotated[dict[Name, Port], Field(max_length=64)]
    outcome: Literal["succeeded", "failed", "unsupported"]
    output: Port | None
    failure: Failure | None

    @model_validator(mode="after")
    def result_shape(self):
        if (self.outcome == "failed") != (self.failure is not None):
            raise ValueError("only failed invocations carry failure")
        if (self.outcome == "failed") != (self.output is None):
            raise ValueError("failed invocation has attempted inputs and no output")
        if (self.kind == "module") != (self.context.training is not None):
            raise ValueError("training mode belongs to each module invocation")
        if (self.kind == "module") != (self.moduleId is not None):
            raise ValueError("module invocations require their instance identity")
        return self


class PortRef(Contract):
    invocationId: Id
    slot: Literal["args", "kwargs", "output"]
    path: Annotated[list[Union[SafeInt, Annotated[StrictStr, Field(max_length=128)]]], Field(max_length=32)]


class TensorEdge(Contract):
    tensorId: Id
    producer: PortRef | None  # null = capture input / unobserved producer, never invented
    consumer: PortRef


class ProcessRunning(Contract):
    state: Literal["running"]


class ProcessExited(Contract):
    state: Literal["exited"]
    exitCode: SafeInt


class ProcessStopped(Contract):
    state: Literal["cancelled", "timed_out", "failed"]
    failure: Failure


ProcessOutcome = Annotated[Union[ProcessRunning, ProcessExited, ProcessStopped], Field(discriminator="state")]


class CaptureOpen(Contract):
    state: Literal["waiting", "recording"]


class CaptureComplete(Contract):
    state: Literal["complete"]
    rootInvocationId: Id


class CaptureIncomplete(Contract):
    state: Literal["incomplete"]
    rootInvocationId: Id | None
    reason: Literal["target_not_observed", "unsupported", "event_limit", "tensor_limit", "byte_limit", "timeout", "cancelled", "recorder_error", "transport_error"]


CaptureOutcome = Annotated[Union[CaptureOpen, CaptureComplete, CaptureIncomplete], Field(discriminator="state")]


class ExecutionRecord(Contract):
    version: Literal[1]
    identity: RunIdentity
    sequence: Nat
    # Host-only RunSpec contains argv/literals. Records safe for views carry identity + limits.
    limits: Limits
    process: ProcessOutcome
    capture: CaptureOutcome
    context: ObservedContext | None
    modules: Annotated[list[ModuleInstance], Field(max_length=100000)]
    invocations: Annotated[list[Invocation], Field(max_length=100000)]
    tensors: Annotated[list[TensorFact], Field(max_length=100000)]
    edges: Annotated[list[TensorEdge], Field(max_length=100000)]
    requirements: Annotated[list[Requirement], Field(max_length=256)]

    @model_validator(mode="after")
    def references(self):
        def index(items):
            result = {item.id: item for item in items}
            if len(result) != len(items):
                raise ValueError("duplicate record id")
            return result
        modules, calls, tensors = index(self.modules), index(self.invocations), index(self.tensors)
        if len(calls) > self.limits.max_events or len(tensors) > self.limits.max_tensors:
            raise ValueError("record exceeds resolved limits")
        seen = set()
        for call in self.invocations:
            if call.parentId is not None and call.parentId not in seen:
                raise ValueError("parent must precede child (cycles forbidden)")
            if call.moduleId is not None and call.moduleId not in modules:
                raise ValueError("unknown module instance")
            seen.add(call.id)
        seen = set()
        object_versions = set()
        for tensor in self.tensors:
            object_version = (tensor.objectId, tensor.version)
            if tensor.version is not None and object_version in object_versions:
                raise ValueError("duplicate tensor object/version identity")
            object_versions.add(object_version)
            for ref in (tensor.aliasOf, tensor.previousVersion):
                if ref is not None and ref not in seen:
                    raise ValueError("tensor alias/version must reference an earlier fact")
            if tensor.aliasOf is not None:
                prior_alias = tensors[tensor.aliasOf]
                if tensor.storageId is not None and prior_alias.storageId is not None and tensor.storageId != prior_alias.storageId:
                    raise ValueError("observed aliases must share storage")
            if tensor.previousVersion is not None:
                prior = tensors[tensor.previousVersion]
                if prior.objectId != tensor.objectId or prior.version is None or tensor.version is None or prior.version >= tensor.version:
                    raise ValueError("previousVersion must advance the same tensor object's version")
            seen.add(tensor.id)

        opaque = False
        def check_port(port, depth=0):
            nonlocal opaque
            if isinstance(port, OpaquePort):
                opaque = True
            if depth > 8:
                raise ValueError("port depth exceeds 8")
            if isinstance(port, TensorPort) and port.tensorId not in tensors:
                raise ValueError("unknown tensor port")
            if isinstance(port, SequencePort):
                for child in port.items:
                    check_port(child, depth + 1)
            if isinstance(port, DictPort):
                keys = [(type(entry.key), entry.key) for entry in port.entries]
                if len(keys) != len(set(keys)):
                    raise ValueError("duplicate dict port key")
                for entry in port.entries:
                    check_port(entry.value, depth + 1)
        for call in self.invocations:
            for port in [*call.inputs, *call.kwargs.values(), call.output]:
                check_port(port)

        def at(ref):
            if ref.invocationId not in calls:
                raise ValueError("edge references unknown invocation")
            call = calls[ref.invocationId]
            port = {"args": call.inputs, "kwargs": call.kwargs, "output": call.output}[ref.slot]
            for part in ref.path:
                if isinstance(port, SequencePort):
                    port = port.items
                elif isinstance(port, DictPort):
                    port = {entry.key: entry.value for entry in port.entries}
                if isinstance(port, list) and type(part) is int and 0 <= part < len(port):
                    port = port[part]
                elif isinstance(port, dict) and part in port:
                    port = port[part]
                else:
                    raise ValueError("edge references missing structured port")
            if not isinstance(port, TensorPort):
                raise ValueError("edge endpoint is not a tensor")
            return port.tensorId
        edge_keys = set()
        consumer_keys = set()
        call_order = {call.id: i for i, call in enumerate(self.invocations)}
        for edge in self.edges:
            if edge.consumer.slot == "output" or (edge.producer and edge.producer.slot != "output"):
                raise ValueError("edges go from outputs to call inputs")
            if at(edge.consumer) != edge.tensorId or (edge.producer and at(edge.producer) != edge.tensorId):
                raise ValueError("edge and port tensor identities differ")
            if edge.producer:
                producer_id, consumer_id = edge.producer.invocationId, edge.consumer.invocationId
                if call_order[producer_id] >= call_order[consumer_id]:
                    raise ValueError("producer must execute before consumer")
                parent = calls[consumer_id].parentId
                while parent is not None:
                    if parent == producer_id:
                        raise ValueError("parent output cannot produce a nested call's input")
                    parent = calls[parent].parentId
            key = edge.model_dump_json()
            if key in edge_keys:
                raise ValueError("duplicate tensor edge")
            edge_keys.add(key)
            consumer_key = edge.consumer.model_dump_json()
            if consumer_key in consumer_keys:
                raise ValueError("a tensor input port cannot have competing producers")
            consumer_keys.add(consumer_key)
        root = getattr(self.capture, "rootInvocationId", None)
        if root is not None and (root not in calls or calls[root].parentId is not None):
            raise ValueError("capture root must be a root invocation")
        if root is not None:
            if calls[root].kind != "module":
                raise ValueError("capture root must be the selected module call")
            if self.context is not None:
                for field in CallContext.model_fields:
                    if getattr(self.context, field) != getattr(calls[root].context, field):
                        raise ValueError("record context must describe the selected root call")
        if self.capture.state == "waiting" and (self.invocations or self.tensors or self.edges or self.context):
            raise ValueError("waiting capture cannot claim invocation evidence")
        if self.capture.state == "complete":
            if (self.context is None or opaque or any(call.outcome == "unsupported" for call in self.invocations)
                    or any(tensor.shape is None for tensor in self.tensors)):
                raise ValueError("complete capture needs observed context and supported invocations")
        if root is not None and any(call.parentId is None and call.id != root for call in self.invocations):
            raise ValueError("record must contain only the selected invocation tree")
        if isinstance(self.capture, CaptureIncomplete) and self.capture.reason == "target_not_observed":
            if root is not None or self.invocations:
                raise ValueError("unobserved target cannot contain captured invocations")
        if self.process.state != "running" and self.capture.state in ("waiting", "recording"):
            raise ValueError("terminal process must close capture")
        if len(self.model_dump_json(exclude_none=False).encode("utf-8")) > self.limits.max_record_bytes:
            raise ValueError("record exceeds byte limit")
        return self


class Freshness(Contract):
    state: Literal["current", "stale"]
    reasons: Annotated[list[Literal["source", "config", "native_file", "profile", "environment", "cancelled", "session", "dependency_scope"]], Field(max_length=8)]

    @model_validator(mode="after")
    def stale_reasons(self):
        if (self.state == "stale") != bool(self.reasons) or len(set(self.reasons)) != len(self.reasons):
            raise ValueError("stale needs distinct reasons; current has none")
        return self


class ExecutionUpdate(Contract):
    version: Literal[1]
    type: Literal["executionUpdate"]
    workspaceSessionId: Id
    record: ExecutionRecord
    freshness: Freshness

    @model_validator(mode="after")
    def same_session(self):
        if self.workspaceSessionId != self.record.identity.workspaceSessionId:
            raise ValueError("foreign execution record")
        return self


class ResultChannel(Contract):
    host: Literal["127.0.0.1"]
    port: Annotated[StrictInt, Field(ge=1, le=65535)]
    token: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}(?![\s\S])")]


class WorkerBootstrap(Contract):
    workspaceRoot: Annotated[StrictStr, Field(min_length=1, max_length=4096)]
    version: Literal[1]
    spec: RunSpec
    resultChannel: ResultChannel

    @model_validator(mode="after")
    def bounded_bootstrap(self):
        if len(self.model_dump_json().encode("utf-8")) > BOOTSTRAP_MAX_BYTES:
            raise ValueError("worker bootstrap exceeds byte limit")
        return self


class WorkerHello(Contract):
    version: Literal[1]
    type: Literal["hello"]
    runId: Id
    token: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}(?![\s\S])")]


class WorkerRecord(Contract):
    dependenciesCurrent: StrictBool
    version: Literal[1]
    type: Literal["record"]
    record: ExecutionRecord


ValidationState = Literal["not_run", "needs_setup", "running", "completed_for_input",
                          "execution_failed", "capture_incomplete", "cancelled", "stale"]

WorkerMessage = Annotated[Union[WorkerHello, WorkerRecord], Field(discriminator="type")]
# Exported roots, not routes. U07 will bind these to authenticated protocol envelopes.
class ValidationContracts(Contract):
    state: ValidationState
    config: ValidationConfig
    resolution: Resolution
    execution: ExecutionUpdate
    worker: WorkerMessage
    bootstrap: WorkerBootstrap


ValidationContracts.model_rebuild()

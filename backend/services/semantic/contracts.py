"""Python-owned semantic capture v1; imports neither Torch nor workspace code.

These are observations, not instructions for materializing or executing inputs.
Validation's binding grammar remains the sole saved-input grammar. JSON paths refer
to the serialized argument tree, so repeated operands never lose their identity.
"""
from __future__ import annotations

import re
import math
import json
from typing import Annotated, Literal, Union

from pydantic import Field, JsonValue, StrictBool, StrictInt, StrictStr, model_validator

from ..validation.contracts import (
    Contract, Digest, FileIdentity, FiniteFloat, Id, Name, Nat, Path,
    Positive, SafeInt, TensorFact, Text, InterpreterIdentity,
)
from ..validation.config import EnvironmentFact

SEMANTIC_VERSION = 1
ArgumentPath = Annotated[list[Union[Nat, Annotated[StrictStr, Field(max_length=128)]]], Field(max_length=64)]
# A boundary tree may be 32 nested dictionaries (3 JSON path components each),
# plus its argument position and record field. Producer paths have a tighter cap.
ProjectionPath = Annotated[list[Union[Nat, Annotated[StrictStr, Field(max_length=128)]]], Field(max_length=100)]
Phase = Literal["parse", "index", "configuration", "environment", "import", "construction",
                "binding", "materialization", "forward", "capture", "projection", "serialization", "transport"]
Omission = Literal["operation_limit", "invocation_limit", "value_limit", "byte_limit", "depth_limit",
                   "item_limit", "unsupported_type", "opaque_kernel", "recorder_error", "timeout", "cancelled"]


class ValueRef(Contract):
    kind: Literal["value"]
    id: Id


class NoneLiteral(Contract):
    kind: Literal["none"]


class BoolLiteral(Contract):
    kind: Literal["bool"]
    value: StrictBool


class IntLiteral(Contract):
    kind: Literal["int"]
    # Decimal text preserves Python integers through JavaScript without rounding.
    value: Annotated[StrictStr, Field(max_length=4096, pattern=r"^(?:0|-?[1-9][0-9]*)(?![\s\S])")]


class FloatLiteral(Contract):
    kind: Literal["float"]
    value: Union[FiniteFloat, Literal["nan", "inf", "-inf", "-0"]]


class TextLiteral(Contract):
    kind: Literal["str", "dtype", "device", "layout", "memory_format"]
    value: Text


class SymbolicArgument(Contract):
    kind: Literal["symbolic"]
    type: Literal["int", "float", "bool"]
    expression: Text


class OpaqueArgument(Contract):
    kind: Literal["opaque"]
    reason: Omission
    # Type name only, never arbitrary object repr, exception text or tensor data.
    typeName: Id | None


class SequenceArgument(Contract):
    kind: Literal["list", "tuple"]
    items: Annotated[list["Argument"], Field(max_length=4096)]


Key = Annotated[Union[NoneLiteral, BoolLiteral, IntLiteral, TextLiteral], Field(discriminator="kind")]


class MappingEntry(Contract):
    key: Key
    value: "Argument"


class MappingArgument(Contract):
    kind: Literal["dict"]
    entries: Annotated[list[MappingEntry], Field(max_length=4096)]

    @model_validator(mode="after")
    def unique_keys(self):
        # Match Python's key equality (True == 1), not just distinct JSON tags.
        keys = [None if isinstance(e.key, NoneLiteral) else int(e.key.value)
                if isinstance(e.key, IntLiteral) else e.key.value for e in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate mapping key")
        return self


Argument = Annotated[Union[ValueRef, NoneLiteral, BoolLiteral, IntLiteral, FloatLiteral,
                          TextLiteral, SymbolicArgument, OpaqueArgument, SequenceArgument,
                          MappingArgument], Field(discriminator="kind")]


def walk_argument(argument, path=(), depth=0):
    """Yield every observation and its exact JSON path, including repeated refs."""
    if depth > 32:
        raise ValueError("argument depth exceeds 32")
    yield path, argument
    if isinstance(argument, SequenceArgument):
        for index, child in enumerate(argument.items):
            yield from walk_argument(child, (*path, "items", index), depth + 1)
    elif isinstance(argument, MappingArgument):
        for index, entry in enumerate(argument.entries):
            yield from walk_argument(entry.value, (*path, "entries", index, "value"), depth + 1)


class Arguments(Contract):
    args: Annotated[list[Argument], Field(max_length=256)]
    kwargs: Annotated[dict[Name, Argument], Field(max_length=256)]

    @model_validator(mode="after")
    def bounded(self):
        count = sum(1 for _ in walk_arguments(self))
        if count > 16384:
            raise ValueError("argument tree exceeds 16384 nodes")
        return self


def walk_arguments(arguments):
    for index, argument in enumerate(arguments.args):
        yield from walk_argument(argument, ("args", index))
    for name, argument in arguments.kwargs.items():
        yield from walk_argument(argument, ("kwargs", name))


class SourceSnapshot(Contract):
    version: Literal[1]
    id: Id
    workspaceSessionId: Id
    workspaceId: Id
    modelId: Id
    modelFile: Path
    className: Name
    generation: Nat
    files: Annotated[list[FileIdentity], Field(min_length=1, max_length=100000)]
    dependencyScope: Literal["complete", "partial"]
    configDigest: Digest | None
    environmentDigest: Digest
    sourceMode: Literal["saved", "buffers"]

    @model_validator(mode="after")
    def identities(self):
        files = {file.path: file for file in self.files}
        if len(files) != len(self.files) or self.modelFile not in files:
            raise ValueError("source snapshot needs distinct files including the selected model")
        if files[self.modelFile].role != "source":
            raise ValueError("selected model must identify source")
        configs = [f for f in self.files if f.role == "config"]
        if self.configDigest is None:
            if configs:
                raise ValueError("config file requires its digest")
        elif len(configs) != 1 or configs[0].path != "tensorviz.toml" or configs[0].digest != self.configDigest:
            raise ValueError("configuration identity must match the workspace-root config")
        return self


class SourceAnchor(Contract):
    file: Path
    digest: Digest
    startLine: Positive
    startColumn: Nat
    endLine: Positive
    endColumn: Nat
    scope: Text
    tokenDigest: Digest

    @model_validator(mode="after")
    def ordered(self):
        if (self.endLine, self.endColumn) <= (self.startLine, self.startColumn):
            raise ValueError("source anchor must cover a nonempty ordered range")
        return self


class SourceBinding(Contract):
    id: Id
    snapshotId: Id
    kind: Literal["definition", "constructor", "call", "argument", "configuration"]
    quality: Literal["exact", "candidate", "ambiguous"]
    anchors: Annotated[list[SourceAnchor], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def exact_anchor(self):
        if self.quality == "exact" and len(self.anchors) != 1:
            raise ValueError("exact binding must have one anchor")
        return self


class Diagnostic(Contract):
    id: Id
    phase: Phase
    code: Literal["invalid_source", "unresolved_import", "invalid_config", "unsaved", "missing_file",
                  "missing_dependency", "unavailable_environment", "unavailable_device", "missing_argument",
                  "unexpected_argument", "invalid_binding", "unknown_signature", "invalid_factory", "invalid_shape", "invalid_dtype",
                  "assertion_failed", "execution_failed", "capture_limited", "timeout", "cancelled",
                  "stale_result", "projection_failed", "serialization_failed", "transport_failed"]
    field: ArgumentPath
    names: Annotated[list[Name], Field(max_length=64)]
    sourceBindingId: Id | None
    expectedShape: Annotated[list[Union[Nat, None]], Field(max_length=32)] | None = None
    actualShape: Annotated[list[Nat], Field(max_length=32)] | None = None
    expectedDtype: Id | None = None
    actualDtype: Id | None = None
    remedy: Literal["fix_source", "save_files", "edit_profile", "select_environment", "review_dependencies",
                    "provide_inputs", "provide_factory", "adjust_limits", "retry", "inspect_failure", "none"]


class ModuleDefinition(Contract):
    id: Id
    qualifiedName: Annotated[StrictStr, Field(min_length=1, max_length=512)]
    sourceBindingIds: Annotated[list[Id], Field(max_length=64)]


InitializerScalar = Annotated[Union[NoneLiteral, BoolLiteral, IntLiteral, FloatLiteral, OpaqueArgument], Field(discriminator="kind")]
InitializerName = Annotated[StrictStr, Field(min_length=1, max_length=128)]


class InitializerParameter(Contract):
    name: InitializerName
    value: InitializerScalar


class InitializerArguments(Contract):
    """Bound scalar parameters at entry, not authored call arguments or completion."""
    phase: Literal["initializer_entry"]
    status: Literal["observed", "unavailable"]
    reason: Literal["byte_limit", "item_limit", "unsupported_type"] | None = None
    positionalOnly: Nat = 0
    positional: Annotated[list[InitializerParameter], Field(max_length=256)] = Field(default=[])
    keywordOnly: Annotated[list[InitializerParameter], Field(max_length=256)] = Field(default=[])
    varArgs: Annotated[list[InitializerScalar], Field(max_length=256)] | None = None
    varKwargs: Annotated[dict[InitializerName, InitializerScalar], Field(max_length=256)] | None = None

    @model_validator(mode="after")
    def bound_parameters(self):
        names = [item.name for item in [*self.positional, *self.keywordOnly]]
        if (len(names) > 256 or len(names) != len(set(names)) or any(not name.isidentifier() for name in names)
                or self.positionalOnly > len(self.positional)):
            raise ValueError("initializer signature needs distinct bounded parameter names")
        if self.varKwargs is not None and set(self.varKwargs) & set(names[self.positionalOnly:]):
            raise ValueError("initializer variadic keywords cannot duplicate named bindings")
        if any(any(0xD800 <= ord(character) <= 0xDFFF for character in name) for name in self.varKwargs or {}):
            raise ValueError("initializer keyword names must be valid Unicode scalars")
        if (self.status == "observed") != (self.reason is None):
            raise ValueError("initializer observation status needs its reason")
        if self.status == "unavailable" and (names or self.positionalOnly or self.varArgs is not None or self.varKwargs is not None):
            raise ValueError("unavailable initializer arguments cannot retain parameter evidence")
        values = [item.value for item in [*self.positional, *self.keywordOnly]] + (self.varArgs or []) + list((self.varKwargs or {}).values())
        if any(isinstance(value, OpaqueArgument) and value.typeName is not None for value in values):
            raise ValueError("unsupported initializer objects retain no object metadata")
        if any(isinstance(value, FloatLiteral) and isinstance(value.value, float) and value.value == 0
               and math.copysign(1, value.value) < 0 for value in values):
            raise ValueError("initializer negative zero needs its explicit string tag")
        return self


class ModuleInstance(Contract):
    id: Id
    definitionId: Id
    # One runtime instance may be registered under several names, including root "".
    # Dynamically created/unregistered modules have no registered path.
    paths: Annotated[list[Text], Field(max_length=4096)]
    # Optional for older captures. Initializer entry is location evidence only;
    # it neither proves successful construction nor authorizes an edit.
    constructorBindingIds: Annotated[list[Id], Field(max_length=64)] = Field(default=[])
    # Optional stronger observation: the saved call's simple name/module path
    # resolved to this exact class at initializer entry, without descriptors.
    constructorClassVerified: StrictBool = False
    initializerArguments: InitializerArguments | None = None

    @model_validator(mode="after")
    def verified_constructor(self):
        if self.constructorClassVerified and len(self.constructorBindingIds) != 1:
            raise ValueError("verified constructor class needs one exact construction site")
        if self.initializerArguments is not None and not self.constructorClassVerified:
            raise ValueError("initializer arguments need verified constructor class identity")
        return self


class OutputRef(Contract):
    operationId: Id
    kind: Literal["output", "effect"]
    path: ArgumentPath


class StateRef(Contract):
    kind: Literal["parameter", "buffer"]
    name: Text


class Value(TensorFact):
    # Separate from TensorFact.version (the actual, sometimes absent torch counter).
    logicalVersion: Nat
    stride: Annotated[list[SafeInt], Field(max_length=32)] | None
    storageOffset: Nat | None
    # Observed numel * element_size, not storage allocation or peak memory.
    logicalBytes: Nat | None = None
    producer: OutputRef | None
    origin: Literal["input", "state", "operation", "external"]
    state: Annotated[list[StateRef], Field(max_length=4096)]

    @model_validator(mode="after")
    def consistent_metadata(self):
        if self.stride is not None and self.shape is not None and len(self.stride) != len(self.shape):
            raise ValueError("stride rank must equal shape rank")
        if (self.origin == "operation") != (self.producer is not None):
            raise ValueError("only operation-produced values have a producer")
        if self.origin == "state" and not self.state:
            raise ValueError("state values identify a parameter or buffer")
        return self


class Invocation(Contract):
    id: Id
    moduleId: Id
    parentId: Id | None
    sequence: Nat
    endSequence: Nat | None
    arguments: Arguments
    output: Argument | None
    status: Literal["completed", "failed", "interrupted"]
    diagnosticId: Id | None
    training: StrictBool
    gradEnabled: StrictBool
    # Recursive unique registered parameters, measured at this call's entry.
    parameterCount: Nat | None = None
    parameterCountUnavailable: Literal["item_limit", "unsupported_type"] | None = None
    callSiteDigest: Digest | None = None
    sourceBindingIds: Annotated[list[Id], Field(max_length=64)]

    @model_validator(mode="after")
    def outcome(self):
        if self.parameterCount is not None and self.parameterCountUnavailable is not None:
            raise ValueError("a measured parameter count cannot also be unavailable")
        if (self.status == "completed") != (self.output is not None):
            raise ValueError("only completed calls have an output tree (including a none literal)")
        if (self.status == "failed") != (self.diagnosticId is not None):
            raise ValueError("failed invocations need a diagnostic")
        if self.endSequence is not None and self.endSequence <= self.sequence:
            raise ValueError("invocation end must follow entry")
        if self.status != "interrupted" and self.endSequence is None:
            raise ValueError("terminal invocation needs an exit sequence")
        return self


class MutationEffect(Contract):
    kind: Literal["write", "alias_update"]
    # The writable argument which caused this change, not a fabricated return.
    argumentPath: ArgumentPath
    beforeId: Id
    after: Annotated[Union[ValueRef, OpaqueArgument], Field(discriminator="kind")]


class Operation(Contract):
    id: Id
    sequence: Nat
    invocationId: Id | None
    target: Annotated[StrictStr, Field(min_length=1, max_length=512)]
    schemaSignature: Annotated[StrictStr, Field(max_length=4096)] | None
    arguments: Arguments
    output: Argument | None
    effects: Annotated[list[MutationEffect], Field(max_length=16384)]
    status: Literal["completed", "failed", "interrupted"]
    diagnosticId: Id | None
    sourceBindingIds: Annotated[list[Id], Field(max_length=64)]

    @model_validator(mode="after")
    def outcome(self):
        if (self.status == "completed") != (self.output is not None):
            raise ValueError("only completed operations have an output tree")
        if (self.status == "failed") != (self.diagnosticId is not None):
            raise ValueError("failed operations need a diagnostic")
        if self.status == "interrupted" and self.effects:
            raise ValueError("interrupted operations cannot claim observed effects")
        return self


class CaptureLimits(Contract):
    # Independent semantic budgets; render depth never limits captured computation.
    operations: Annotated[Positive, Field(le=100000)]
    invocations: Annotated[Positive, Field(le=100000)]
    values: Annotated[Positive, Field(le=100000)]
    bytes: Annotated[Positive, Field(ge=1024, le=67108864)]


def _saved_environment(source, interpreter, environment):
    if source.sourceMode != "saved":
        raise ValueError("execution requires a saved source snapshot")
    if interpreter.environmentDigest != source.environmentDigest:
        raise ValueError("execution interpreter differs from source applicability")
    from .execution_plan import fingerprint
    if (environment.executable != interpreter.executable
            or environment.python_version != interpreter.version
            or fingerprint(environment.model_dump(mode="json")) != source.environmentDigest):
        raise ValueError("execution environment facts differ from their identity")


class AutomaticCaptureRequest(Contract):
    """Existing automatic/declared-input request, not a resolved input recipe."""
    input: Annotated[StrictStr, Field(min_length=1, max_length=4096)] | None
    tensorInputs: Annotated[list[dict[StrictStr, JsonValue]], Field(max_length=8)] | None
    overrides: dict[StrictStr, JsonValue] | None

    @model_validator(mode="after")
    def bounded(self):
        from ..ingest.tensor_inputs import validate_tensor_inputs
        from ..validation.setup import _literals
        if self.input is not None and self.tensorInputs is not None:
            raise ValueError("choose one input description")
        if self.tensorInputs is not None:
            validate_tensor_inputs(self.tensorInputs)
        if self.overrides is not None:
            _literals(self.overrides)
        return self


class CapturePreparation(Contract):
    """Saved facts before automatic inputs resolve; cannot impersonate a run plan."""
    version: Literal[1]
    id: Id
    source: SourceSnapshot
    interpreter: InterpreterIdentity
    environment: EnvironmentFact
    configuration: Literal["absent", "no_match"]
    policy: Literal["legacy-input-search-v1"]
    request: AutomaticCaptureRequest
    timeoutMs: Annotated[StrictInt, Field(ge=100, le=86400000)]
    captureLimits: CaptureLimits

    @model_validator(mode="after")
    def prepared(self):
        _saved_environment(self.source, self.interpreter, self.environment)
        if (self.configuration == "absent") != (self.source.configDigest is None):
            raise ValueError("configuration discovery must match the source snapshot")
        return self


class ResolvedExecutionPlan(Contract):
    """Host-resolved saved invocation. ExecutionSpec remains the recipe authority.

    Automatic guesses are not resolved plans. The recipe's nested inputs retain
    validation.Binding or explicit tensor/fixture descriptions, never observations.
    """
    version: Literal[1]
    id: Id
    source: SourceSnapshot
    interpreter: InterpreterIdentity
    environment: EnvironmentFact
    origin: Literal["profile", "explicit"]
    configuration: Literal["absent", "selected", "overridden"]
    profileName: Name | None
    profileDigest: Digest | None
    execution: dict[StrictStr, JsonValue]
    job: dict[StrictStr, JsonValue] | None
    timeoutMs: Annotated[StrictInt, Field(ge=100, le=86400000)]
    captureLimits: CaptureLimits

    @model_validator(mode="after")
    def resolved(self):
        from ..ingest.execution_spec import validate_execution_spec, validate_execution_job
        self.execution = validate_execution_spec(self.execution)
        if self.job is not None:
            self.job = validate_execution_job(self.job)
        _saved_environment(self.source, self.interpreter, self.environment)
        if self.origin == "profile":
            if self.configuration != "selected" or self.profileName is None or self.profileDigest is None:
                raise ValueError("selected profiles require configuration and profile identities")
        elif self.profileName is not None or self.profileDigest is not None or self.configuration == "selected":
            raise ValueError("explicit execution cannot claim a selected profile")
        if (self.configuration == "absent") != (self.source.configDigest is None):
            raise ValueError("configuration discovery must match the source snapshot")
        files = {f.path: f for f in self.source.files}
        expected = []
        if "setup" in self.execution:
            expected.append((self.execution["setup"]["file"], "source"))
        if "fixture" in self.execution["inputs"]:
            expected.append((self.execution["inputs"]["fixture"], "native"))
        if "checkpoint" in self.execution:
            expected.append((self.execution["checkpoint"]["file"], "native"))
        if any(path not in files or files[path].role != role for path, role in expected):
            raise ValueError("execution files require matching saved identities")
        return self


class CandidateExecution(Contract):
    """Private replay provenance; never a replacement for the live model capture."""
    version: Literal[1]
    id: Id
    baseCaptureId: Id
    baseAttemptId: Id
    baseSourceId: Id
    baseWorkspaceId: Id
    basePlanDigest: Digest
    intentDigest: Digest
    file: Path
    beforeDigest: Digest
    afterDigest: Digest
    plan: ResolvedExecutionPlan

    @model_validator(mode="after")
    def distinct_candidate(self):
        if (self.id != self.plan.id or self.id == self.baseAttemptId
                or self.plan.source.id == self.baseSourceId or self.plan.source.workspaceId == self.baseWorkspaceId):
            raise ValueError("candidate replay requires its own attempt and source identity")
        file = next((f for f in self.plan.source.files if f.path == self.file), None)
        if file is None or file.role != "source" or file.digest != self.afterDigest:
            raise ValueError("candidate replay must contain the exact proposed source")
        if self.beforeDigest == self.afterDigest:
            raise ValueError("unchanged source is not a candidate edit")
        return self


class CoverageEntry(Contract):
    modulePath: Text
    status: Literal["observed", "failed", "not_observed", "capture_limited", "static_only", "unresolved", "state"]
    invocationIds: Annotated[list[Id], Field(max_length=100000)]
    reason: Omission | None

    @model_validator(mode="after")
    def evidence(self):
        if (self.status in ("observed", "failed")) != bool(self.invocationIds):
            raise ValueError("observed/failed coverage needs invocation evidence; other states cannot claim calls")
        if (self.status == "capture_limited") != (self.reason is not None):
            raise ValueError("capture-limited coverage needs its omission reason")
        return self


class CaptureCoverage(Contract):
    scope: Literal["executed_path", "traced_program"]
    operationsSeen: Nat
    invocationsSeen: Nat
    omissions: Annotated[list[Omission], Field(max_length=16)]
    modules: Annotated[list[CoverageEntry], Field(max_length=100000)]


class SourceCorrespondence(Contract):
    """Revision-specific evidence for a source card, not proof that an expression ran."""
    nodeId: Text
    captureId: Id
    sourceId: Id
    status: Literal["observed", "failed", "not_observed", "capture_limited", "static_only", "unresolved", "state"]
    basis: Literal["registered_path", "shared_instance", "state_reference", "no_mapping"]
    moduleId: Id | None
    invocationIds: Annotated[list[Id], Field(max_length=100000)]
    stateValueIds: Annotated[list[Id], Field(max_length=100000)]
    reason: Omission | None

    @model_validator(mode="after")
    def evidence(self):
        if (len(set(self.invocationIds)) != len(self.invocationIds)
                or len(set(self.stateValueIds)) != len(self.stateValueIds)):
            raise ValueError("source correspondence references must be unique")
        if (self.status in ("observed", "failed")) != bool(self.invocationIds):
            raise ValueError("observed source correspondence needs invocation evidence")
        if (self.status == "capture_limited") != (self.reason is not None):
            raise ValueError("limited source correspondence needs a capture reason")
        if self.basis in ("registered_path", "shared_instance"):
            if self.moduleId is None or self.stateValueIds or self.status in ("state", "static_only", "unresolved"):
                raise ValueError("module correspondence needs a registered instance")
        elif self.moduleId is not None or self.invocationIds:
            raise ValueError("unmapped/state correspondence cannot claim a module call")
        if (self.basis == "state_reference") != (self.status == "state"):
            raise ValueError("state correspondence requires state evidence")
        if self.status != "state" and self.stateValueIds:
            raise ValueError("only state correspondence can cite state values")
        if self.basis == "no_mapping" and self.status not in ("capture_limited", "static_only", "unresolved"):
            raise ValueError("unmapped source correspondence cannot establish a module instance")
        return self


class ProjectionNode(Contract):
    id: Id
    kind: Literal["operation", "invocation", "input", "output", "value"]
    semanticId: Id
    parentId: Id | None
    visible: StrictBool
    collapsed: StrictBool
    hiddenOperationCount: Nat


class ProjectionPort(Contract):
    id: Id
    nodeId: Id
    direction: Literal["in", "out"]
    # Paths are relative to this node's captured record. An input value can have
    # several names/positions; those aliases are kept, not arbitrarily selected.
    paths: Annotated[list[ProjectionPath], Field(max_length=16384)]
    argumentKind: Literal["value", "none", "bool", "int", "float", "str", "dtype", "device",
                          "layout", "memory_format", "symbolic", "opaque", "list", "tuple", "dict"]
    valueId: Id | None
    # A collapsed interface port cites its exact hidden member port.
    memberPortId: Id | None

    @model_validator(mode="after")
    def binding(self):
        if (self.argumentKind == "value") != (self.valueId is not None):
            raise ValueError("tensor ports must identify their observed value")
        if len({tuple(path) for path in self.paths}) != len(self.paths):
            raise ValueError("port paths must be unique")
        if self.memberPortId is not None and self.paths:
            raise ValueError("a collapsed boundary refers to its member instead of inventing a local path")
        return self


class ProjectionEdge(Contract):
    id: Id
    kind: Literal["argument", "effect", "result"]
    valueId: Id
    sourcePortId: Id
    targetPortId: Id
    # Null together for a connection hidden wholly inside one collapsed region.
    # Original endpoints always survive; display endpoints never replace them.
    displaySourcePortId: Id | None
    displayTargetPortId: Id | None


class CaptureReference(Contract):
    version: Literal[1]
    captureId: Id
    sourceId: Id
    kind: Literal["operation", "invocation", "input", "output", "value"]
    semanticId: Id


class SourceFloatLiteral(Contract):
    kind: Literal["float"]
    value: Union[FiniteFloat, Literal["-0"]]

    @model_validator(mode="after")
    def signed_zero(self):
        if isinstance(self.value, float) and self.value == 0 and math.copysign(1, self.value) < 0:
            raise ValueError("source negative zero needs its explicit string tag")
        return self


SourceNumber = Annotated[Union[IntLiteral, SourceFloatLiteral], Field(discriminator="kind")]


class SourceEditScope(Contract):
    # One observed event is never permission to specialize a shared expression.
    kind: Literal["source_expression"]
    affects: Literal["all_executions"]
    observedOperations: Positive
    observedInvocations: Nat
    observedModuleInstances: Nat
    includesUnobservedExecutions: Literal[True]


class LiteralEditCapability(Contract):
    version: Literal[1]
    id: Id
    command: Literal["set_literal"]
    recipe: Literal["binary_scalar_v1"]
    target: CaptureReference
    expression: SourceBinding
    binding: SourceBinding
    argumentPath: ArgumentPath
    expected: SourceNumber
    scope: SourceEditScope

    @model_validator(mode="after")
    def exact_source(self):
        if self.target.kind != "operation" or self.argumentPath != ["args", 1]:
            raise ValueError("binary scalar recipe requires the exact second operation argument")
        for binding, kind in ((self.expression, "call"), (self.binding, "argument")):
            if binding.snapshotId != self.target.sourceId or binding.kind != kind or binding.quality != "exact":
                raise ValueError("literal recipe needs exact bindings in its base snapshot")
        outer, inner = self.expression.anchors[0], self.binding.anchors[0]
        if ((outer.file, outer.digest, outer.scope) != (inner.file, inner.digest, inner.scope)
                or (inner.startLine, inner.startColumn) < (outer.startLine, outer.startColumn)
                or (inner.endLine, inner.endColumn) > (outer.endLine, outer.endColumn)):
            raise ValueError("literal binding must be inside its authored expression")
        return self


class SetLiteralEdit(Contract):
    kind: Literal["set_literal"]
    capabilityId: Id
    target: CaptureReference
    argumentPath: ArgumentPath
    expected: SourceNumber
    value: SourceNumber
    scope: Literal["source_expression"]


class SourceValueChoice(Contract):
    name: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    value: ValueRef
    binding: SourceBinding

    @model_validator(mode="after")
    def identifier(self):
        if not self.name.isidentifier() or self.binding.kind != "argument" or self.binding.quality != "exact":
            raise ValueError("source values need a proven name definition")
        return self


def _binary_argument_path(path):
    return len(path) == 2 and path[0] == "args" and type(path[1]) is int and path[1] in (0, 1)


class ReconnectArgumentCapability(Contract):
    version: Literal[1]
    id: Id
    command: Literal["reconnect_argument"]
    recipe: Literal["straight_line_tensor_v1"]
    target: CaptureReference
    expression: SourceBinding
    binding: SourceBinding
    argumentPath: ArgumentPath
    expected: ValueRef
    expectedName: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    choices: Annotated[list[SourceValueChoice], Field(min_length=1, max_length=128)]
    scope: SourceEditScope

    @model_validator(mode="after")
    def exact_source(self):
        if self.target.kind != "operation" or not _binary_argument_path(self.argumentPath) or not self.expectedName.isidentifier():
            raise ValueError("reconnection needs an exact named binary operand")
        for binding, kind in ((self.expression, "call"), (self.binding, "argument")):
            if binding.snapshotId != self.target.sourceId or binding.kind != kind or binding.quality != "exact":
                raise ValueError("reconnection needs exact source bindings")
        outer, inner = self.expression.anchors[0], self.binding.anchors[0]
        if ((outer.file, outer.digest, outer.scope) != (inner.file, inner.digest, inner.scope)
                or (inner.startLine, inner.startColumn) < (outer.startLine, outer.startColumn)
                or (inner.endLine, inner.endColumn) > (outer.endLine, outer.endColumn)):
            raise ValueError("reconnection operand must be inside its expression")
        if (len({choice.value.id for choice in self.choices}) != len(self.choices)
                or not any(choice.value == self.expected and choice.name == self.expectedName for choice in self.choices)):
            raise ValueError("available values must be unique and retain the original operand")
        for choice in self.choices:
            anchor = choice.binding.anchors[0]
            if (choice.binding.snapshotId != self.target.sourceId
                    or (anchor.file, anchor.digest, anchor.scope) != (outer.file, outer.digest, outer.scope)
                    or (anchor.endLine, anchor.endColumn) > (outer.startLine, outer.startColumn)):
                raise ValueError("available value definitions must precede the expression in its scope")
        return self


class ReconnectArgumentEdit(Contract):
    kind: Literal["reconnect_argument"]
    capabilityId: Id
    target: CaptureReference
    argumentPath: ArgumentPath
    expected: ValueRef
    value: ValueRef
    scope: Literal["source_expression"]

    @model_validator(mode="after")
    def operand(self):
        if self.target.kind != "operation" or not _binary_argument_path(self.argumentPath):
            raise ValueError("reconnection needs an exact operation argument")
        return self


def _constructor_argument_path(path):
    return (len(path) == 2 and ((path[0] == "args" and type(path[1]) is int and 0 <= path[1] < 256)
            or (path[0] == "kwargs" and type(path[1]) is str and path[1].isidentifier())))


class ConstructorEditScope(SourceEditScope):
    observedOperations: Nat
    observedInvocations: Positive
    observedModuleInstances: Positive


class ConstructorLiteralCapability(Contract):
    version: Literal[1]
    id: Id
    command: Literal["set_constructor_literal"]
    recipe: Literal["constructor_scalar_v1"]
    target: CaptureReference
    expression: SourceBinding
    binding: SourceBinding
    # This path indexes the authored constructor call, not forward arguments or
    # a dispatch schema. No inferred constructor parameter name is promised.
    sourceArgumentPath: ArgumentPath
    constructor: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    expected: SourceNumber
    scope: ConstructorEditScope

    @model_validator(mode="after")
    def exact_source(self):
        if self.target.kind != "invocation" or not _constructor_argument_path(self.sourceArgumentPath):
            raise ValueError("constructor scalar recipe requires a source argument and module invocation")
        for binding, kind in ((self.expression, "constructor"), (self.binding, "argument")):
            if binding.snapshotId != self.target.sourceId or binding.kind != kind or binding.quality != "exact":
                raise ValueError("constructor scalar recipe needs exact source bindings")
        outer, inner = self.expression.anchors[0], self.binding.anchors[0]
        if ((outer.file, outer.digest, outer.scope) != (inner.file, inner.digest, inner.scope)
                or (inner.startLine, inner.startColumn) < (outer.startLine, outer.startColumn)
                or (inner.endLine, inner.endColumn) > (outer.endLine, outer.endColumn)):
            raise ValueError("constructor argument must be inside its authored call")
        return self


class SetConstructorLiteralEdit(Contract):
    kind: Literal["set_constructor_literal"]
    capabilityId: Id
    target: CaptureReference
    sourceArgumentPath: ArgumentPath
    expected: SourceNumber
    value: SourceNumber
    scope: Literal["source_expression"]

    @model_validator(mode="after")
    def source_argument(self):
        if self.target.kind != "invocation" or not _constructor_argument_path(self.sourceArgumentPath):
            raise ValueError("constructor edits require the exact source argument, not a captured argument")
        return self


class ConfigurationEditScope(Contract):
    kind: Literal["configuration_definition"]
    affects: Literal["all_references"]
    knownSourceReferences: Positive
    observedConstructorSites: Positive
    observedOperations: Nat
    observedInvocations: Positive
    observedModuleInstances: Positive
    includesUnobservedReferences: Literal[True]


class ConfigurationLiteralCapability(Contract):
    version: Literal[1]
    id: Id
    command: Literal["set_configuration_literal"]
    recipe: Literal["configuration_scalar_v1"]
    target: CaptureReference
    expression: SourceBinding
    reference: SourceBinding
    binding: SourceBinding
    sourceArgumentPath: ArgumentPath
    symbol: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    expected: SourceNumber
    scope: ConfigurationEditScope

    @model_validator(mode="after")
    def exact_definition(self):
        if self.target.kind != "invocation" or not _constructor_argument_path(self.sourceArgumentPath) or not self.symbol.isidentifier():
            raise ValueError("configuration edits require a named constructor argument reference")
        for binding, kind in ((self.expression, "constructor"), (self.reference, "argument"), (self.binding, "configuration")):
            if binding.snapshotId != self.target.sourceId or binding.kind != kind or binding.quality != "exact":
                raise ValueError("configuration edits require exact source bindings")
        outer, ref, definition = (b.anchors[0] for b in (self.expression, self.reference, self.binding))
        if (len({(a.file, a.digest) for a in (outer, ref, definition)}) != 1
                or ref.scope != outer.scope or definition.scope not in (outer.scope, "<module>")
                or (ref.startLine, ref.startColumn) < (outer.startLine, outer.startColumn)
                or (ref.endLine, ref.endColumn) > (outer.endLine, outer.endColumn)
                or (definition.endLine, definition.endColumn) > (outer.startLine, outer.startColumn)):
            raise ValueError("configuration definition and constructor reference need a consistent source scope")
        return self


class SetConfigurationLiteralEdit(Contract):
    kind: Literal["set_configuration_literal"]
    capabilityId: Id
    target: CaptureReference
    sourceArgumentPath: ArgumentPath
    expected: SourceNumber
    value: SourceNumber
    scope: Literal["configuration_definition"]

    @model_validator(mode="after")
    def source_reference(self):
        if self.target.kind != "invocation" or not _constructor_argument_path(self.sourceArgumentPath):
            raise ValueError("configuration edits require an invocation and exact source reference path")
        return self


ActivationName = Literal["relu", "gelu", "silu", "sigmoid", "tanh"]


class ActivationImportPlan(Contract):
    module: Literal["torch.nn", "torch.nn.functional"]
    line: Positive
    lineEnding: Literal["\n", "\r\n"]


class ActivationSourceOptions(Contract):
    # None means absent from source, not an inferred runtime default.
    inplace: StrictBool | None = None
    approximate: Literal["none", "tanh"] | None = None


class ActivationEditCapability(Contract):
    version: Literal[1]
    id: Id
    command: Literal["replace_activation"]
    recipe: Literal["functional_activation_v1", "module_activation_v1"]
    target: CaptureReference
    expression: SourceBinding
    binding: SourceBinding
    namespace: Annotated[StrictStr, Field(min_length=1, max_length=256,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?![\s\S])")]
    importPlan: ActivationImportPlan | None = None
    expected: ActivationName
    options: ActivationSourceOptions = Field(default_factory=ActivationSourceOptions)
    compatibleActivations: Annotated[list[ActivationName], Field(min_length=1, max_length=5)] = Field(
        default_factory=lambda: ["relu", "gelu", "silu", "sigmoid", "tanh"])
    scope: SourceEditScope

    @model_validator(mode="after")
    def exact_source(self):
        compatible = (["relu", "silu"] if self.options.inplace is not None else
                      ["gelu"] if self.options.approximate is not None else ["relu", "gelu", "silu", "sigmoid", "tanh"])
        if (self.compatibleActivations != compatible or self.expected not in compatible
                or self.options.inplace is not None and self.options.approximate is not None):
            raise ValueError("activation choices must preserve the authored options")
        module = self.recipe == "module_activation_v1"
        if self.target.kind != ("invocation" if module else "operation"):
            raise ValueError("activation recipe requires its exact operation or module invocation")
        for binding, kind in ((self.expression, "constructor" if module else "call"), (self.binding, "call")):
            if binding.snapshotId != self.target.sourceId or binding.kind != kind or binding.quality != "exact":
                raise ValueError("activation recipe needs exact expression bindings in its base snapshot")
        outer, inner = self.expression.anchors[0], self.binding.anchors[0]
        if ((outer.file, outer.digest, outer.scope) != (inner.file, inner.digest, inner.scope)
                or (inner.startLine, inner.startColumn) < (outer.startLine, outer.startColumn)
                or (inner.endLine, inner.endColumn) > (outer.endLine, outer.endColumn)):
            raise ValueError("callee binding must be inside its authored expression")
        if self.importPlan is not None and (self.importPlan.module != ("torch.nn" if module else "torch.nn.functional")
                or self.importPlan.line > outer.startLine or not self.namespace.isidentifier()
                or not self.namespace.startswith("_tensorviz_")):
            raise ValueError("activation import must use its bounded namespace before the expression")
        return self


class ReplaceActivationEdit(Contract):
    kind: Literal["replace_activation"]
    capabilityId: Id
    target: CaptureReference
    expected: ActivationName
    value: ActivationName
    scope: Literal["source_expression"]


class InsertUnaryCapability(Contract):
    version: Literal[1]
    id: Id
    command: Literal["insert_unary_op"]
    recipe: Literal["straight_line_unary_insert_v1"]
    target: CaptureReference
    expression: SourceBinding
    binding: SourceBinding
    argumentPath: ArgumentPath
    expected: ValueRef
    expectedName: Annotated[StrictStr, Field(min_length=1, max_length=128)]
    namespace: Annotated[StrictStr, Field(min_length=1, max_length=256,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?![\s\S])")]
    importPlan: ActivationImportPlan | None = None
    scope: SourceEditScope

    @model_validator(mode="after")
    def exact_source(self):
        if self.target.kind != "operation" or not _binary_argument_path(self.argumentPath) or not self.expectedName.isidentifier():
            raise ValueError("unary insertion needs an exact named binary operand")
        for binding, kind in ((self.expression, "call"), (self.binding, "argument")):
            if binding.snapshotId != self.target.sourceId or binding.kind != kind or binding.quality != "exact":
                raise ValueError("unary insertion needs exact source bindings")
        outer, inner = self.expression.anchors[0], self.binding.anchors[0]
        if ((outer.file, outer.digest, outer.scope) != (inner.file, inner.digest, inner.scope)
                or (inner.startLine, inner.startColumn) < (outer.startLine, outer.startColumn)
                or (inner.endLine, inner.endColumn) > (outer.endLine, outer.endColumn)):
            raise ValueError("unary insertion must wrap only its bound operand")
        if self.importPlan is not None and (self.importPlan.module != "torch.nn.functional"
                or self.importPlan.line > outer.startLine or not self.namespace.isidentifier()
                or not self.namespace.startswith("_tensorviz_")):
            raise ValueError("unary insertion needs its declared functional import")
        return self


class InsertUnaryEdit(Contract):
    kind: Literal["insert_unary_op"]
    capabilityId: Id
    target: CaptureReference
    argumentPath: ArgumentPath
    expected: ValueRef
    value: ActivationName
    scope: Literal["source_expression"]

    @model_validator(mode="after")
    def operand(self):
        if self.target.kind != "operation" or not _binary_argument_path(self.argumentPath):
            raise ValueError("unary insertion needs an exact operation argument")
        return self


class DeleteUnaryCapability(Contract):
    version: Literal[1]
    id: Id
    command: Literal["delete_unary_op"]
    recipe: Literal["straight_line_unary_delete_v1"]
    target: CaptureReference
    expression: SourceBinding
    binding: SourceBinding
    argumentPath: ArgumentPath
    expected: ValueRef
    activation: ActivationName
    input: SourceValueChoice
    scope: SourceEditScope

    @model_validator(mode="after")
    def exact_source(self):
        if (self.target.kind != "operation" or not _binary_argument_path(self.argumentPath) or self.argumentPath[1] != 0
                or self.expected == self.input.value or self.binding != self.expression
                or self.binding.snapshotId != self.target.sourceId or self.binding.kind != "call" or self.binding.quality != "exact"):
            raise ValueError("unary deletion needs its exact call and distinct input/output values")
        outer, definition = self.expression.anchors[0], self.input.binding.anchors[0]
        if (self.input.binding.snapshotId != self.target.sourceId
                or (outer.file, outer.digest, outer.scope) != (definition.file, definition.digest, definition.scope)
                or (definition.endLine, definition.endColumn) > (outer.startLine, outer.startColumn)
                or outer.startLine != outer.endLine):
            raise ValueError("unary deletion needs one call and its preceding input definition")
        return self


class DeleteUnaryEdit(Contract):
    kind: Literal["delete_unary_op"]
    capabilityId: Id
    target: CaptureReference
    argumentPath: ArgumentPath
    expected: ValueRef
    value: ValueRef
    scope: Literal["source_expression"]

    @model_validator(mode="after")
    def operand(self):
        if self.target.kind != "operation" or not _binary_argument_path(self.argumentPath) or self.argumentPath[1] != 0:
            raise ValueError("unary deletion needs an exact unary input")
        return self


class DisplayPort(ProjectionPort):
    shape: Annotated[list[Nat], Field(max_length=32)] | None
    dtype: Id | None
    device: Id | None

    @model_validator(mode="after")
    def measured(self):
        if self.argumentKind != "value" and any(value is not None for value in (self.shape, self.dtype, self.device)):
            raise ValueError("only value ports can have tensor metadata")
        return self


class CaptureProjection(Contract):
    version: Literal[1]
    captureId: Id
    sourceId: Id
    collapsedInvocationIds: Annotated[list[Id], Field(max_length=100000)]
    nodes: Annotated[list[ProjectionNode], Field(max_length=300002)]
    ports: Annotated[list[ProjectionPort], Field(max_length=1000000)]
    edges: Annotated[list[ProjectionEdge], Field(max_length=1000000)]

    @model_validator(mode="after")
    def references(self):
        nodes, ports, edges = map(_index, (self.nodes, self.ports, self.edges))
        seen = set()
        hidden_by = {}
        counts = {id: 0 for id in nodes}
        for node in self.nodes:
            if node.parentId is not None:
                if node.parentId not in seen or nodes[node.parentId].kind != "invocation":
                    raise ValueError("projection ownership needs an earlier invocation")
                parent = nodes[node.parentId]
                hidden_by[node.id] = hidden_by[parent.id] or (parent.id if parent.collapsed else None)
            else:
                hidden_by[node.id] = None
            if node.visible != (hidden_by[node.id] is None):
                raise ValueError("visibility must follow invocation collapse")
            if node.kind != "invocation" and (node.collapsed or node.hiddenOperationCount):
                raise ValueError("only invocation regions can collapse")
            counts[node.id] = int(node.kind == "operation")
            seen.add(node.id)
        for node in reversed(self.nodes):
            if node.parentId is not None:
                counts[node.parentId] += counts[node.id]
            if node.hiddenOperationCount != (counts[node.id] if node.collapsed else 0):
                raise ValueError("hidden operation count must count retained descendants")
        collapsed = [n.semanticId for n in self.nodes if n.collapsed]
        if self.collapsedInvocationIds != collapsed:
            raise ValueError("collapse identities must match their invocation nodes")
        for port in self.ports:
            if port.nodeId not in nodes:
                raise ValueError("port needs a known node")
            if port.memberPortId is not None:
                member = ports.get(port.memberPortId)
                if (member is None or member.memberPortId is not None
                        or hidden_by[member.nodeId] != port.nodeId
                        or (member.direction, member.valueId, member.argumentKind) !=
                           (port.direction, port.valueId, port.argumentKind)):
                    raise ValueError("collapsed port must cite its hidden member's exact value and direction")
        for edge in self.edges:
            endpoints = []
            for id, direction in ((edge.sourcePortId, "out"), (edge.targetPortId, "in")):
                port = ports.get(id)
                if port is None or port.direction != direction or port.valueId != edge.valueId or port.memberPortId is not None:
                    raise ValueError("connection needs real value evidence at both original ports")
                endpoints.append(port)
            source, target = endpoints
            owner = hidden_by[source.nodeId]
            hidden = owner is not None and owner == hidden_by[target.nodeId]
            for original, display in ((source, edge.displaySourcePortId), (target, edge.displayTargetPortId)):
                if hidden:
                    if display is not None:
                        raise ValueError("internal collapsed connections must remain hidden")
                elif display not in ports or not nodes[ports[display].nodeId].visible or (
                        display != original.id and ports[display].memberPortId != original.id):
                    raise ValueError("visible endpoint must preserve its original port or exact collapsed boundary")
        return self


def _index(items):
    result = {item.id: item for item in items}
    if len(result) != len(items):
        raise ValueError("duplicate semantic identity")
    return result


class AutomaticCaptureSelection(Contract):
    provenance: Literal["explicit", "declared", "seed-from-first-layer", "search-candidate"]
    candidateIndex: Annotated[Positive, Field(le=80)]
    candidatesTried: Annotated[Positive, Field(le=80)]

    @model_validator(mode="after")
    def selected(self):
        if self.candidateIndex > self.candidatesTried:
            raise ValueError("selected candidate must have been tried")
        return self


class CaptureSnapshot(Contract):
    version: Literal[1]
    id: Id
    attemptId: Id
    source: SourceSnapshot
    planDigest: Digest | None
    preparationDigest: Digest | None
    automatic: AutomaticCaptureSelection | None
    inputDigest: Digest | None
    stateDigest: Digest | None
    # "fake_tensor": an observed value or registered parameter/buffer was a FakeTensor.
    # Values were never materialized: Value facts keep shape, dtype, device, stride and
    # identity; value digests (inputDigest/stateDigest of fake tensors) and statistics
    # do not exist. Omitted, not null, for real tensors, so real captures are unchanged.
    # The schema declares no default either: a validator that fills defaults (the
    # host's portable-export check uses Ajv useDefaults) would otherwise add
    # "valuesUnavailable": null to every real capture it checks.
    valuesUnavailable: Literal["fake_tensor"] | None = Field(
        default=None, exclude_if=lambda value: value is None,
        json_schema_extra=lambda schema: schema.pop("default", None))
    backend: Literal["runtime", "export", "jaxpr"]
    backendVersion: Id
    torchVersion: Id | None
    jaxVersion: Id | None = None
    normalizations: Annotated[list[Text], Field(max_length=64)]
    status: Literal["complete", "partial", "failed"]
    rootInvocationId: Id | None
    inputs: Arguments
    output: Argument | None
    definitions: Annotated[list[ModuleDefinition], Field(max_length=100000)]
    modules: Annotated[list[ModuleInstance], Field(max_length=100000)]
    invocations: Annotated[list[Invocation], Field(max_length=100000)]
    operations: Annotated[list[Operation], Field(max_length=100000)]
    values: Annotated[list[Value], Field(max_length=100000)]
    sourceBindings: Annotated[list[SourceBinding], Field(max_length=100000)]
    diagnostics: Annotated[list[Diagnostic], Field(max_length=4096)]
    coverage: CaptureCoverage
    limits: CaptureLimits

    @model_validator(mode="after")
    def relationships(self):
        if self.backend == 'jaxpr':
            if self.jaxVersion is None or self.torchVersion is not None or self.coverage.scope != 'traced_program':
                raise ValueError('Jaxpr capture needs JAX version and traced-program coverage')
        elif self.coverage.scope != 'executed_path' or self.torchVersion is None or self.jaxVersion is not None:
            raise ValueError('PyTorch capture needs executed-path coverage and Torch version')
        if (self.planDigest is None) == (self.preparationDigest is None):
            raise ValueError("capture needs exactly one plan or preparation identity")
        if (self.preparationDigest is not None) != (self.automatic is not None):
            raise ValueError("automatic captures need their selected candidate identity")
        if self.source.sourceMode != "saved":
            raise ValueError("runtime capture currently requires saved source; buffer overlays are not implemented")
        definitions, modules, calls, ops, values, bindings, diagnostics = map(_index, (
            self.definitions, self.modules, self.invocations, self.operations, self.values,
            self.sourceBindings, self.diagnostics))
        files = {file.path: file for file in self.source.files}
        for binding in self.sourceBindings:
            if binding.snapshotId != self.source.id:
                raise ValueError("binding belongs to another source snapshot")
            if any(a.file not in files or a.digest != files[a.file].digest for a in binding.anchors):
                raise ValueError("binding anchor must match its exact file digest")
        for item in [*self.definitions, *self.invocations, *self.operations]:
            if len(item.sourceBindingIds) != len(set(item.sourceBindingIds)) or any(b not in bindings for b in item.sourceBindingIds):
                raise ValueError("unknown or duplicate source binding")
        for diagnostic in self.diagnostics:
            if diagnostic.sourceBindingId is not None and diagnostic.sourceBindingId not in bindings:
                raise ValueError("unknown diagnostic source binding")
        paths = set()
        for module in self.modules:
            if module.definitionId not in definitions or any(p in paths for p in module.paths) or len(set(module.paths)) != len(module.paths):
                raise ValueError("module needs a known definition and distinct instance paths")
            if (len(module.constructorBindingIds) != len(set(module.constructorBindingIds))
                    or any(b not in bindings or bindings[b].kind != "constructor" for b in module.constructorBindingIds)):
                raise ValueError("module constructor references need distinct constructor bindings")
            if module.constructorClassVerified and bindings[module.constructorBindingIds[0]].quality != "exact":
                raise ValueError("verified constructor class needs an exact construction binding")
            paths.update(module.paths)
        seen = set()
        sequences = set()
        for call in self.invocations:
            if call.moduleId not in modules or (call.parentId is not None and call.parentId not in seen):
                raise ValueError("invocation needs a known module and an earlier parent")
            if call.diagnosticId is not None and call.diagnosticId not in diagnostics:
                raise ValueError("unknown invocation diagnostic")
            if call.parentId is not None:
                parent = calls[call.parentId]
                if parent.sequence >= call.sequence or (parent.endSequence is not None and
                        (call.endSequence is None or call.endSequence >= parent.endSequence)):
                    raise ValueError("child invocation must fit inside its parent")
            for seq in [call.sequence] + ([] if call.endSequence is None else [call.endSequence]):
                if seq in sequences:
                    raise ValueError("duplicate event sequence")
                sequences.add(seq)
            seen.add(call.id)
        previous = -1
        for op in self.operations:
            if op.sequence <= previous or op.sequence in sequences:
                raise ValueError("operations must have unique ascending event sequences")
            previous = op.sequence
            sequences.add(op.sequence)
            if op.invocationId is not None:
                if op.invocationId not in calls:
                    raise ValueError("unknown containing invocation")
                call = calls[op.invocationId]
                if not call.sequence < op.sequence or (call.endSequence is not None and op.sequence >= call.endSequence):
                    raise ValueError("operation must execute inside its invocation")
            if op.diagnosticId is not None and op.diagnosticId not in diagnostics:
                raise ValueError("unknown operation diagnostic")
        opaque = False

        def check_tree(tree, before=None):
            nonlocal opaque
            count = 0
            for _, argument in tree:
                count += 1
                if count > 16384:
                    raise ValueError("argument tree exceeds 16384 nodes")
                if isinstance(argument, OpaqueArgument):
                    opaque = True
                if isinstance(argument, ValueRef):
                    if argument.id not in values:
                        raise ValueError("unknown value reference")
                    producer = values[argument.id].producer
                    if producer is not None and (producer.operationId not in ops or
                            (before is not None and ops[producer.operationId].sequence >= before)):
                        raise ValueError("value producer must precede its use")

        check_tree(walk_arguments(self.inputs), before=-1)
        for call in self.invocations:
            check_tree(walk_arguments(call.arguments), before=call.sequence)
            if call.output is not None:
                check_tree(walk_argument(call.output), before=call.endSequence)
        for op in self.operations:
            check_tree(walk_arguments(op.arguments), before=op.sequence)
            if op.output is not None:
                check_tree(walk_argument(op.output), before=op.sequence + 1)
            arguments = dict(walk_arguments(op.arguments))
            effect_keys = set()
            for effect in op.effects:
                key = (effect.kind, tuple(effect.argumentPath), effect.beforeId)
                if key in effect_keys:
                    raise ValueError("duplicate mutation effect")
                effect_keys.add(key)
                argument = arguments.get(tuple(effect.argumentPath))
                if not isinstance(argument, ValueRef) or effect.beforeId not in values:
                    raise ValueError("mutation effect needs a known writable argument and prior value")
                before = values[effect.beforeId]
                if effect.kind == "write" and effect.beforeId != argument.id:
                    raise ValueError("write effect must identify its exact argument value")
                if effect.kind == "alias_update" and (before.id == argument.id or before.storageId is None
                        or before.storageId != values[argument.id].storageId):
                    raise ValueError("alias effect must share its writable argument's storage")
                check_tree([((), ValueRef(kind="value", id=effect.beforeId))], before=op.sequence)
                check_tree([((), effect.after)], before=op.sequence + 1)
                if isinstance(effect.after, ValueRef):
                    after = values[effect.after.id]
                    if after.previousVersion != before.id or after.producer is None or after.producer.operationId != op.id:
                        raise ValueError("mutation effect must advance its prior value through this operation")
        if self.output is not None:
            check_tree(walk_argument(self.output))
        seen = set()
        object_versions = set()
        object_latest = {}
        outputs = {op.id: dict(walk_argument(op.output)) for op in self.operations if op.output is not None}
        effects = {op.id: {(i, "after"): effect.after for i, effect in enumerate(op.effects)} for op in self.operations}
        for value in self.values:
            key = (value.objectId, value.logicalVersion)
            if key in object_versions:
                raise ValueError("duplicate object/logical-version")
            object_versions.add(key)
            for ref in (value.aliasOf, value.previousVersion):
                if ref is not None and ref not in seen:
                    raise ValueError("alias/version must reference an earlier value")
            if value.aliasOf is not None:
                alias = values[value.aliasOf]
                if value.storageId is not None and alias.storageId is not None and value.storageId != alias.storageId:
                    raise ValueError("aliased values must share storage")
            if value.previousVersion is not None:
                prior = values[value.previousVersion]
                if (prior.objectId != value.objectId or prior.logicalVersion + 1 != value.logicalVersion
                        or object_latest.get(value.objectId) != prior.id
                        or (prior.version is not None and value.version is not None and prior.version > value.version)):
                    raise ValueError("previous value must advance the same object's version")
            elif value.logicalVersion != 0 or value.objectId in object_latest:
                raise ValueError("logical versions need the previous observed value of this object")
            if value.producer is not None:
                ports = outputs if value.producer.kind == "output" else effects
                port = ports.get(value.producer.operationId, {}).get(tuple(value.producer.path))
                if not isinstance(port, ValueRef) or port.id != value.id:
                    raise ValueError("producer output path must reference the produced value")
            seen.add(value.id)
            object_latest[value.objectId] = value.id
        root = calls.get(self.rootInvocationId)
        if (self.rootInvocationId is not None and root is None) or (root is not None and root.parentId is not None):
            raise ValueError("capture root must identify a root invocation")
        if any(c.parentId is None and c.id != self.rootInvocationId for c in self.invocations):
            raise ValueError("capture must contain one selected invocation tree")
        if root is not None and (root.arguments != self.inputs or root.output != self.output):
            raise ValueError("capture boundaries must match the selected invocation")
        coverage = self.coverage
        if coverage.operationsSeen < len(ops) or coverage.invocationsSeen < len(calls):
            raise ValueError("seen event counts cannot be smaller than retained evidence")
        if ((coverage.operationsSeen != len(ops) or coverage.invocationsSeen != len(calls)) and not coverage.omissions):
            raise ValueError("missing events need an explicit omission reason")
        if len(coverage.omissions) != len(set(coverage.omissions)):
            raise ValueError("duplicate omission reason")
        coverage_paths = set()
        calls_by_path = {}
        for call in self.invocations:
            for path in modules[call.moduleId].paths:
                calls_by_path.setdefault(path, set()).add(call.id)
        for entry in coverage.modules:
            if entry.modulePath in coverage_paths:
                raise ValueError("duplicate coverage path")
            coverage_paths.add(entry.modulePath)
            if len(entry.invocationIds) != len(set(entry.invocationIds)):
                raise ValueError("duplicate coverage invocation")
            if set(entry.invocationIds) != calls_by_path.get(entry.modulePath, set()):
                raise ValueError("coverage cannot omit observed invocations or claim observed modules were not called")
            for call_id in entry.invocationIds:
                if call_id not in calls or entry.modulePath not in modules[calls[call_id].moduleId].paths:
                    raise ValueError("coverage must cite calls of the specified module")
            if entry.status == "failed" and not any(calls[i].status == "failed" for i in entry.invocationIds):
                raise ValueError("failed coverage needs a failed invocation")
        if self.status == "complete" and (root is None or root.status != "completed" or opaque or coverage.omissions
                or coverage.operationsSeen != len(ops) or coverage.invocationsSeen != len(calls)
                or any(c.status == "interrupted" for c in self.invocations)
                or any(o.status == "interrupted" or (o.status == "failed" and o.effects) for o in self.operations)):
            raise ValueError("complete capture cannot hide omitted, unsupported or unfinished evidence")
        if len(ops) > self.limits.operations or len(calls) > self.limits.invocations or len(values) > self.limits.values:
            raise ValueError("capture exceeds semantic budget")
        if len(self.model_dump_json().encode("utf-8")) > self.limits.bytes:
            raise ValueError("capture exceeds byte budget")
        return self


class PortableCapture(Contract):
    # The portable envelope is separate from both runtime and graph-file versions.
    version: Literal[1]
    sourceNames: Literal["anonymized"]
    capture: CaptureSnapshot

    @model_validator(mode="after")
    def detached_source(self):
        source = self.capture.source
        if (source.workspaceSessionId != "portable" or source.workspaceId != "portable"
                or source.modelId != "portable" or source.generation != 0
                or re.fullmatch(r"portable:[0-9a-f]{64}", source.id) is None
                or any(file.documentVersion is not None for file in source.files)
                or [file.path for file in source.files] != ["tensorviz.toml" if file.role == "config"
                    else f"file-{index}.py" for index, file in enumerate(source.files)]):
            raise ValueError("portable captures must detach live source names and editor identity")
        local_path = re.compile(r"^(?:/|[a-z]:[\\/]|\\\\|file:|vscode:)", re.I)
        private_name = re.compile(r"^(?:_.*|source|sourceCode|source_code|code|file|file_path|path|location|token|session|sessionId|credential|secret|api[_-]?key|authorization|password)$", re.I)

        def check(argument, private=False):
            if argument is None or isinstance(argument, (ValueRef, NoneLiteral)):
                return
            if isinstance(argument, SequenceArgument):
                for child in argument.items:
                    check(child, private)
                return
            if isinstance(argument, MappingArgument):
                for entry in argument.entries:
                    if isinstance(entry.key, TextLiteral) and local_path.match(entry.key.value):
                        raise ValueError("captured arguments contain a private path; lossless export refused")
                    check(entry.value, private or bool(isinstance(entry.key, TextLiteral) and private_name.match(entry.key.value)))
                return
            if private or isinstance(argument, TextLiteral) and local_path.match(argument.value):
                raise ValueError("captured arguments contain a private literal; lossless export refused")

        for record in [self.capture, *self.capture.invocations, *self.capture.operations]:
            arguments = record.inputs if isinstance(record, CaptureSnapshot) else record.arguments
            for argument in arguments.args:
                check(argument)
            for name, argument in arguments.kwargs.items():
                check(argument, bool(private_name.match(name)))
            check(record.output)
        for module in self.capture.modules:
            entry = module.initializerArguments
            if entry is None:
                continue
            # These closed scalar snapshots contain no strings/object metadata.
            # A leading underscore on a declared numeric parameter (e.g.
            # Embedding._freeze) is not itself a private payload. Sensitive
            # names still refuse; opaque observations contain no value to leak.
            def scalar_check(value, private=False):
                if not isinstance(value, OpaqueArgument):
                    check(value, private)
            for parameter in [*entry.positional, *entry.keywordOnly]:
                scalar_check(parameter.value, bool(private_name.match(parameter.name.lstrip("_"))))
            for value in entry.varArgs or []:
                scalar_check(value)
            for name, value in (entry.varKwargs or {}).items():
                if local_path.match(name):
                    raise ValueError("initializer arguments contain a private path; lossless export refused")
                scalar_check(value, bool(private_name.match(name)))
        return self


class RunAttempt(Contract):
    version: Literal[1]
    id: Id
    origin: Literal["manual", "mcp", "automatic"]
    requestDigest: Digest
    # Setup may refuse before an executable plan exists. Never invent one for a
    # configuration error or an automatic input guess.
    sourceId: Id | None
    planDigest: Digest | None
    phase: Phase
    status: Literal["requested", "running", "needs_setup", "succeeded", "failed", "cancelled", "timed_out"]
    startedAt: Nat
    finishedAt: Nat | None
    captureIds: Annotated[list[Id], Field(max_length=8)]
    diagnostics: Annotated[list[Diagnostic], Field(max_length=256)]
    # Execution cache (`on` mode): no worker ran; this attempt reused an earlier
    # attempt's identical capture, which keeps that attempt's ids (capture, source,
    # bindings). Null for every attempt that executed.
    reusedCaptureId: Id | None = None

    @model_validator(mode="after")
    def terminal(self):
        if self.reusedCaptureId is not None and self.reusedCaptureId not in self.captureIds:
            raise ValueError("a reused capture must be one of the attempt's captures")
        if self.planDigest is not None and self.sourceId is None:
            raise ValueError("resolved plan requires source identity")
        if (self.status not in ("requested", "running")) != (self.finishedAt is not None):
            raise ValueError("only terminal attempts have a finish time")
        if self.finishedAt is not None and self.finishedAt < self.startedAt:
            raise ValueError("attempt cannot finish before it starts")
        if len(set(self.captureIds)) != len(self.captureIds):
            raise ValueError("duplicate attempt capture")
        return self


class CaptureState(Contract):
    latestAttempt: RunAttempt | None
    lastCompleteCaptureId: Id | None
    lastPartialCaptureId: Id | None
    displayedCaptureId: Id | None


class CandidateRequest(Contract):
    """Private host payload; a closed recipe plus the full reviewed artifact."""
    version: Literal[1]
    id: Id
    plan: ResolvedExecutionPlan
    capture: CaptureSnapshot
    intent: Annotated[Union[SetLiteralEdit, ReplaceActivationEdit, SetConstructorLiteralEdit,
        SetConfigurationLiteralEdit, ReconnectArgumentEdit, InsertUnaryEdit, DeleteUnaryEdit],
        Field(discriminator="kind")]
    reviewedFile: Path
    reviewedSource: Annotated[StrictStr, Field(max_length=1048576)]


class CaptureReconciliationRequest(Contract):
    """Private static check of a later live capture against the approved edit."""
    version: Literal[1]
    base: CaptureSnapshot
    sources: Annotated[dict[Path, StrictStr], Field(max_length=128)]
    intent: Annotated[Union[SetLiteralEdit, ReplaceActivationEdit, SetConstructorLiteralEdit,
        SetConfigurationLiteralEdit, ReconnectArgumentEdit, InsertUnaryEdit, DeleteUnaryEdit],
        Field(discriminator="kind")]
    reviewedFile: Path
    reviewedSource: Annotated[StrictStr, Field(max_length=1048576)]
    current: CaptureSnapshot


class ConfigurationArgumentUse(Contract):
    bindingId: Id
    sourceArgumentPath: ArgumentPath

    @model_validator(mode="after")
    def argument(self):
        if not _constructor_argument_path(self.sourceArgumentPath):
            raise ValueError("configuration evidence needs an exact constructor argument")
        return self


class CandidatePostconditions(Contract):
    """Evidence for this captured expression only, never program equivalence."""
    version: Literal[1]
    executionId: Id
    captureId: Id | None
    recipe: Literal["binary_scalar_v1", "functional_activation_v1", "module_activation_v1", "straight_line_tensor_v1",
                    "straight_line_unary_insert_v1", "straight_line_unary_delete_v1", "constructor_scalar_v1", "configuration_scalar_v1"] | None
    scope: Literal["observed_expression", "observed_initializer_entries", "observed_configuration_arguments"]
    status: Literal["verified", "failed", "unavailable"]
    reason: Literal["observed_requested_literal", "observed_literal_mismatch",
                    "observed_requested_activation", "observed_activation_mismatch", "unsupported_recipe",
                    "observed_requested_argument", "observed_argument_mismatch",
                    "observed_requested_insertion", "observed_insertion_mismatch",
                    "observed_requested_deletion", "observed_deletion_mismatch",
                    "observed_requested_constructor", "observed_constructor_mismatch", "initializer_arguments_unavailable",
                    "observed_requested_configuration", "observed_configuration_mismatch", "configuration_reference_unavailable",
                    "capture_unavailable", "capture_incomplete", "capture_identity_mismatch",
                    "source_mapping_unavailable", "expression_not_observed", "ambiguous_observation"]
    bindingId: Id | None
    observedOperations: Annotated[Nat, Field(le=100000)]
    observedInitializers: Annotated[Nat, Field(le=100000)] = 0
    configurationUses: Annotated[list[ConfigurationArgumentUse], Field(max_length=256)] = Field(default=[])

    @model_validator(mode="after")
    def evidence(self):
        checked = self.status != "unavailable"
        configuration = self.recipe == "configuration_scalar_v1"
        constructor = self.recipe == "constructor_scalar_v1"
        initializers = constructor or configuration
        if (self.scope != ("observed_configuration_arguments" if configuration else "observed_initializer_entries" if constructor else "observed_expression")
                or (self.observedOperations if initializers else self.observedInitializers)):
            raise ValueError("postcondition scope and counts must describe their evidence")
        uses = [(use.bindingId, tuple(use.sourceArgumentPath)) for use in self.configurationUses]
        if ((bool(uses) != (checked and configuration)) or len(set(uses)) != len(uses)
                or uses and self.bindingId not in {binding for binding, _path in uses}
                or len(json.dumps([use.model_dump(mode="json") for use in self.configurationUses], ensure_ascii=False,
                    separators=(",", ":")).encode()) > 32768):
            raise ValueError("configuration checks need bounded unique constructor argument witnesses")
        count = self.observedInitializers if initializers else self.observedOperations
        name = ("literal" if self.recipe == "binary_scalar_v1" else
                "configuration" if configuration else
                "constructor" if constructor else
                "argument" if self.recipe == "straight_line_tensor_v1" else
                "insertion" if self.recipe == "straight_line_unary_insert_v1" else
                "deletion" if self.recipe == "straight_line_unary_delete_v1" else "activation")
        expected = {"verified": f"observed_requested_{name}", "failed": f"observed_{name}_mismatch"}
        checked_reasons = {"observed_requested_literal", "observed_literal_mismatch", "observed_requested_activation", "observed_activation_mismatch",
                           "observed_requested_argument", "observed_argument_mismatch",
                           "observed_requested_insertion", "observed_insertion_mismatch",
                           "observed_requested_deletion", "observed_deletion_mismatch",
                           "observed_requested_constructor", "observed_constructor_mismatch",
                           "observed_requested_configuration", "observed_configuration_mismatch"}
        if checked != (self.reason in checked_reasons) or checked and self.reason != expected[self.status]:
            raise ValueError("postcondition status must agree with its evidence")
        if checked and (self.recipe is None or self.captureId is None or self.bindingId is None or count == 0):
            raise ValueError("checked postconditions need captured expression evidence")
        if not checked and (self.bindingId is not None or count != 0):
            raise ValueError("unavailable postconditions cannot claim checked observations")
        if (self.recipe is None) != (self.reason == "unsupported_recipe"):
            raise ValueError("unsupported recipes remain explicit")
        return self


class CaptureReconciliation(Contract):
    """Observed source round trip, never freshness certification of an editor buffer."""
    version: Literal[1]
    baseCaptureId: Id
    captureId: Id
    sourceId: Id
    reviewedDigest: Digest
    dependencyScope: Literal["complete", "partial"]
    status: Literal["checked", "unavailable"]
    reason: Literal["reviewed_source", "source_reverted", "source_changed", "dependencies_changed",
        "environment_changed", "different_model", "capture_not_new", "source_unavailable"]
    inputs: Literal["same", "different", "unavailable"]
    state: Literal["same", "different", "unavailable"]
    postconditions: CandidatePostconditions | None

    @model_validator(mode="after")
    def evidence(self):
        checked = self.status == "checked"
        if checked != (self.reason == "reviewed_source") or checked != (self.postconditions is not None):
            raise ValueError("only exact reviewed source can have semantic reconciliation")
        if not checked and (self.inputs != "unavailable" or self.state != "unavailable"):
            raise ValueError("unavailable reconciliation cannot compare execution conditions")
        if self.postconditions is not None and self.postconditions.captureId != self.captureId:
            raise ValueError("reconciliation must describe the selected live capture")
        return self


class SemanticContracts(Contract):
    capture: CaptureSnapshot
    portableCapture: PortableCapture
    state: CaptureState
    plan: ResolvedExecutionPlan
    candidateExecution: CandidateExecution
    captureReconciliationRequest: CaptureReconciliationRequest
    captureReconciliation: CaptureReconciliation
    candidateRequest: CandidateRequest
    candidatePostconditions: CandidatePostconditions
    preparation: CapturePreparation
    correspondence: SourceCorrespondence
    projection: CaptureProjection
    captureReference: CaptureReference
    displayPort: DisplayPort
    literalEditCapability: LiteralEditCapability
    setLiteralEdit: SetLiteralEdit
    reconnectArgumentCapability: ReconnectArgumentCapability
    reconnectArgumentEdit: ReconnectArgumentEdit
    activationEditCapability: ActivationEditCapability
    replaceActivationEdit: ReplaceActivationEdit
    insertUnaryCapability: InsertUnaryCapability
    insertUnaryEdit: InsertUnaryEdit
    deleteUnaryCapability: DeleteUnaryCapability
    deleteUnaryEdit: DeleteUnaryEdit
    constructorLiteralCapability: ConstructorLiteralCapability
    setConstructorLiteralEdit: SetConstructorLiteralEdit
    configurationLiteralCapability: ConfigurationLiteralCapability
    setConfigurationLiteralEdit: SetConfigurationLiteralEdit


SemanticContracts.model_rebuild()

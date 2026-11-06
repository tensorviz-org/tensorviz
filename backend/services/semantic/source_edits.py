"""Exact authored edit recipes over a retained capture and supplied source buffers.

This module never reads files, imports a model, evaluates an expression, or writes
source. Locations and runtime argument positions are distinct; each recipe must
prove their relationship before preparing a patch. The host still owns freshness,
native review and application.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
import math
import io
import tokenize

from ..torch_imports import dotted_name as _dotted_name, torch_imports
from ..validation.source import MAX_BYTES, MAX_FILE_BYTES, MAX_FILES
from .contracts import (
    CaptureSnapshot, CaptureReference, LiteralEditCapability, SetLiteralEdit,
    IntLiteral, FloatLiteral, SourceFloatLiteral, ValueRef,
    ActivationEditCapability, ReplaceActivationEdit,
    ConstructorLiteralCapability, SetConstructorLiteralEdit,
    ConfigurationLiteralCapability, SetConfigurationLiteralEdit,
    ReconnectArgumentCapability, ReconnectArgumentEdit,
    InsertUnaryCapability, InsertUnaryEdit, DeleteUnaryCapability, DeleteUnaryEdit,
)
from .source_bindings import MAX_SITES, SourceIndex, source_binding
from .source_configuration import ConfigurationIndex
from .source_activations import (TARGETS as _ACTIVATION_TARGETS, CLASSES as _ACTIVATION_CLASSES,
    INPLACE_TARGETS, source_options, observed_activation, observed_activation_call)
from .source_flow import BINARY_TARGETS as _BINARY_TARGETS, StraightLineFlow, FlowRefused
from .construction import constructor_binding


class SourceEditRefused(ValueError):
    """Safe, closed refusal code; never include source or exception bodies."""


_TORCH_IMPORT_TARGETS = {"torch.nn", "torch.nn.functional"} | {
    "torch.nn.functional." + name for name in _ACTIVATION_TARGETS
} | {"torch.nn." + name for name in _ACTIVATION_CLASSES.values()}


def _torch_imports(tree):
    return torch_imports(tree, _TORCH_IMPORT_TARGETS)


def _number(node):
    sign = 1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -1 if isinstance(node.op, ast.USub) else 1
        node = node.operand
    if not isinstance(node, ast.Constant) or type(node.value) not in (int, float):
        return None
    value = sign * node.value
    if type(value) is int:
        return IntLiteral(kind="int", value=str(value)) if value.bit_length() <= 13600 else None
    if not math.isfinite(value):
        return None
    return SourceFloatLiteral(kind="float", value="-0" if value == 0 and math.copysign(1, value) < 0 else value)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def observed_binary_scalar(operation, target):
    """The bounded observation grammar shared by editing and candidate checks."""
    args, kwargs = operation.arguments.args, operation.arguments.kwargs
    if (operation.status != "completed" or operation.effects or len(operation.sourceBindingIds) != 1
            or operation.target != target or len(args) != 2 or not isinstance(args[0], ValueRef)
            or not isinstance(args[1], (IntLiteral, FloatLiteral))
            or kwargs and not (target in ("aten.add.Tensor", "aten.sub.Tensor")
                and list(kwargs) == ["alpha"] and kwargs["alpha"].model_dump(mode="json") == {"kind": "int", "value": "1"})):
        return None
    return args[1]


class CapturedEditResolver:
    """Bounded source proof shared by capability discovery and proposal planning."""

    source_number = staticmethod(_number)

    def __init__(self, payload, sources):
        self.capture = CaptureSnapshot.model_validate(payload)
        if isinstance(payload, CaptureSnapshot):
            self.capture = self.capture.model_copy(deep=True)
        # Host snapshots may include every Python file in a large workspace.
        # Only the model and recorded binding files can participate in edits;
        # apply the proof budgets and digest checks to that complete subset.
        referenced = {self.capture.source.modelFile}
        referenced.update(anchor.file for binding in self.capture.sourceBindings for anchor in binding.anchors)
        self.sources = {path: text for path, text in sources.items() if path in referenced}
        self.files = {file.path: file for file in self.capture.source.files if file.role == "source"}
        if len(self.sources) > MAX_FILES:
            raise SourceEditRefused("source_limit")
        total = 0
        for path, source in self.sources.items():
            if type(source) is not str:
                raise SourceEditRefused("source_unavailable")
            data = source.encode("utf-8")
            total += len(data)
            if len(data) > MAX_FILE_BYTES or total > MAX_BYTES:
                raise SourceEditRefused("source_limit")
            if path not in self.files or hashlib.sha256(data).hexdigest() != self.files[path].digest:
                raise SourceEditRefused("stale_source")
        self.bindings = {binding.id: binding for binding in self.capture.sourceBindings}
        self.operations = {op.id: op for op in self.capture.operations}
        self.values = {value.id: value for value in self.capture.values}
        self.calls = {call.id: call for call in self.capture.invocations}
        self.modules = {module.id: module for module in self.capture.modules}
        self.definitions = {definition.id: definition for definition in self.capture.definitions}
        self.by_constructor, self.by_module = defaultdict(list), defaultdict(list)
        self.by_invocation, self.child_calls = defaultdict(list), defaultdict(list)
        for module in self.capture.modules:
            for binding_id in module.constructorBindingIds:
                self.by_constructor[binding_id].append(module)
        for call in self.capture.invocations:
            self.by_module[call.moduleId].append(call)
            self.child_calls[call.parentId].append(call)
        self.by_expression = defaultdict(list)
        for op in self.capture.operations:
            self.by_invocation[op.invocationId].append(op)
            for binding_id in op.sourceBindingIds:
                self.by_expression[binding_id].append(op)
        self.indexes = {}
        self.namespaces = {}
        self.new_imports = {}
        self.configuration_indexes, self.configuration_scopes = {}, {}
        self.source_flows = {}
        self.remaining_sites = MAX_SITES

    def _expression(self, binding):
        if binding.quality != "exact" or binding.kind not in ("call", "constructor"):
            raise SourceEditRefused("ambiguous_binding")
        anchor = binding.anchors[0]
        file = anchor.file
        if file not in self.sources:
            raise SourceEditRefused("source_unavailable")
        if file not in self.indexes:
            if self.remaining_sites <= 0:
                raise SourceEditRefused("source_limit")
            try:
                index = SourceIndex(self.capture.source.id, self.files[file], self.sources[file].encode(),
                    max_sites=self.remaining_sites, retain_nodes=True)
            except (ValueError, SyntaxError, RecursionError, UnicodeError):
                raise SourceEditRefused("source_limit") from None
            self.indexes[file] = index
            self.remaining_sites -= index.site_count
        key = binding.id.removeprefix("constructor:") if binding.kind == "constructor" else binding.id
        found = self.indexes[file].bound_nodes.get(key)
        expected = constructor_binding(found[0]) if found is not None and binding.kind == "constructor" else found[0] if found else None
        if found is None or expected != binding.model_dump(mode="json"):
            raise SourceEditRefused("source_binding_mismatch")
        return found[1]

    def navigation(self, binding_ids):
        if len(binding_ids) != 1:
            return None
        binding = self.bindings.get(binding_ids[0])
        if binding is None:
            return None
        try:
            self._expression(binding)
        except SourceEditRefused:
            return None
        anchor = binding.anchors[0]
        return dict(file=anchor.file, start=[anchor.startLine, anchor.startColumn],
            end=[anchor.endLine, anchor.endColumn], kind="call", digest=anchor.tokenDigest,
            file_digest=anchor.digest)

    def capability(self, operation_id):
        if operation_id in self.calls:
            return self.module_activation_capability(operation_id)
        op = self.operations.get(operation_id)
        if op is not None and op.target in (*_ACTIVATION_TARGETS.values(), *INPLACE_TARGETS.values()):
            return self.activation_capability(operation_id)
        return self.literal_capability(operation_id)

    def capabilities(self, semantic_id):
        """Publish every supported field without changing single-recipe callers."""
        if semantic_id not in self.calls:
            available, primary_error = [], None
            try:
                available.append(self.capability(semantic_id))
            except SourceEditRefused as error:
                primary_error = error
            try:
                available.extend(self.reconnect_capabilities(semantic_id))
                try:
                    available.extend(self.insert_unary_capabilities(semantic_id))
                except SourceEditRefused:
                    pass  # Namespace proof can refuse insertion while reconnection stays valid.
            except SourceEditRefused as error:
                if not available:
                    raise primary_error if str(error) == "unsupported_reconnection" else error
            if available and available[0].command == "replace_activation":
                try:
                    available.append(self.delete_unary_capability(semantic_id))
                except SourceEditRefused:
                    pass  # Replacement does not require the stronger straight-line bypass proof.
            return available
        try:
            return [self.module_activation_capability(semantic_id)]
        except SourceEditRefused as activation_error:
            try:
                numeric = self.constructor_literal_capabilities(semantic_id)
            except SourceEditRefused as constructor_error:
                if constructor_error.args != ("no_constructor_literals",):
                    raise
                numeric = []
            try:
                numeric.extend(self.configuration_literal_capabilities(semantic_id))
            except SourceEditRefused as configuration_error:
                if configuration_error.args != ("no_configuration_literals",):
                    raise
            if numeric:
                def position(capability):
                    binding = capability.reference if isinstance(capability, ConfigurationLiteralCapability) else capability.binding
                    anchor = binding.anchors[0]
                    return anchor.startLine, anchor.startColumn
                return sorted(numeric, key=position)
            module = self.modules[self.calls[semantic_id].moduleId]
            name = self.definitions[module.definitionId].qualifiedName
            if name.rsplit(".", 1)[-1] in _ACTIVATION_CLASSES.values():
                raise activation_error
            raise SourceEditRefused("no_constructor_literals")

    def _activation_name(self, callee, anchor, *, module=False):
        """Prove the existing call, then choose a safe replacement namespace.

        Direct aliases keep their import and other uses intact. Only a changed
        selected callee uses the new/reused namespace; a no-op adds no import.
        """
        if anchor.file not in self.namespaces:
            self.namespaces[anchor.file] = _torch_imports(self.indexes[anchor.file].tree)
        imports = self.namespaces[anchor.file]
        spelling = _dotted_name(callee)
        if spelling is None:
            raise SourceEditRefused("unproven_activation_import")
        root, _, tail = spelling.partition(".")
        imported = imports.get(root)
        target = imported + ("." + tail if tail else "") if imported else None
        parent = "torch.nn" if module else "torch.nn.functional"
        names = _ACTIVATION_CLASSES if module else {name: name for name in _ACTIVATION_TARGETS}
        expected = next((name for name, symbol in names.items() if target == parent + "." + symbol), None)
        if expected is None:
            raise SourceEditRefused("unproven_activation_import")
        if isinstance(callee, ast.Attribute):
            return expected, _dotted_name(callee.value), None
        namespace, plan = self._torch_namespace(anchor, parent)
        return expected, namespace, plan

    def _torch_namespace(self, anchor, parent="torch.nn.functional"):
        if anchor.file not in self.namespaces:
            self.namespaces[anchor.file] = _torch_imports(self.indexes[anchor.file].tree)
        imports = self.namespaces[anchor.file]
        namespaces = [name + parent[len(target):] for name, target in imports.items()
                      if target == parent or parent.startswith(target + ".")]
        if namespaces:
            return min(namespaces, key=lambda name: (len(name), name)), None
        key = (anchor.file, parent)
        if key in self.new_imports:
            return self.new_imports[key]
        source = self.sources[anchor.file]
        # Python tokenizer NAME tokens include bindings in patterns, imports,
        # global/nonlocal declarations and scopes, without evaluating anything.
        used = {token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.NAME}
        base = "_tensorviz_nn" if parent == "torch.nn" else "_tensorviz_functional"
        namespace = base
        for suffix in range(1, len(used) + 2):
            if namespace not in used:
                break
            namespace = base + "_" + str(suffix)
        body = self.indexes[anchor.file].tree.body
        boundary = 0
        for index, statement in enumerate(body):
            docstring = index == 0 and isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and type(statement.value.value) is str
            if docstring or isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
                boundary = statement.end_lineno
                continue
            if statement.col_offset != 0 or statement.lineno <= boundary:
                raise SourceEditRefused("unsafe_import_insertion")
            newline = source.find("\n")
            plan = dict(module=parent, line=statement.lineno,
                lineEnding="\r\n" if newline > 0 and source[newline - 1] == "\r" else "\n")
            self.new_imports[key] = (namespace, plan)
            return namespace, plan
        raise SourceEditRefused("unsafe_import_insertion")

    def plan(self, payload):
        if not isinstance(payload, dict):
            raise SourceEditRefused("unsupported_edit")
        planner = {"set_literal": self.plan_literal, "replace_activation": self.plan_activation,
                   "set_constructor_literal": self.plan_constructor_literal,
                   "set_configuration_literal": self.plan_configuration_literal,
                   "reconnect_argument": self.plan_reconnection,
                   "insert_unary_op": self.plan_insert_unary,
                   "delete_unary_op": self.plan_delete_unary}.get(payload.get("kind"))
        if planner is None:
            raise SourceEditRefused("unsupported_edit")
        return planner(payload)

    def reconnect_capabilities(self, operation_id):
        op = self.operations.get(operation_id)
        if op is None:
            raise SourceEditRefused("unsupported_target")
        if len(op.sourceBindingIds) != 1:
            raise SourceEditRefused("ambiguous_binding")
        expression = self.bindings[op.sourceBindingIds[0]]
        node = self._expression(expression)
        if not isinstance(node, ast.BinOp) or type(node.op) not in _BINARY_TARGETS:
            raise SourceEditRefused("unsupported_reconnection")
        flow = self._source_flow(op, expression, node)
        state = flow.states[op.invocationId]
        available = {name: item for name, item in state.items() if len(name) <= 128 and isinstance(item.value, ValueRef)
            and all(name in other and isinstance(other[name].value, ValueRef) and other[name].definition is item.definition
                    for other in flow.states.values())}
        scope = dict(kind="source_expression", affects="all_executions", observedOperations=len(flow.events),
            observedInvocations=len(flow.states), observedModuleInstances=len({self.calls[id].moduleId for id in flow.states}),
            includesUnobservedExecutions=True)
        capabilities = []
        for index, operand in enumerate((node.left, node.right)):
            if not isinstance(operand, ast.Name) or operand.id not in available:
                continue
            expected = available[operand.id].value
            if expected != op.arguments.args[index]:
                raise SourceEditRefused("ambiguous_source_flow")
            choices, seen = [], set()
            # Prefer the original spelling for an alias of the same observed value.
            for name in [operand.id, *available]:
                item = available[name]
                if item.value.id in seen:
                    continue
                binding = flow.binding(item.definition)
                if binding is None:
                    raise SourceEditRefused("source_flow_limit")
                choices.append(dict(name=name, value=item.value.model_dump(mode="json"), binding=binding))
                seen.add(item.value.id)
            binding = flow.binding(operand)
            if binding is None:
                raise SourceEditRefused("source_flow_limit")
            facts = dict(version=1, command="reconnect_argument", recipe="straight_line_tensor_v1",
                target=CaptureReference(version=1, captureId=self.capture.id, sourceId=self.capture.source.id,
                    kind="operation", semanticId=op.id).model_dump(mode="json"),
                expression=expression.model_dump(mode="json"), binding=binding, argumentPath=["args", index],
                expected=expected.model_dump(mode="json"), expectedName=operand.id, choices=choices, scope=scope)
            capabilities.append(ReconnectArgumentCapability(id="edit:" + _identity(facts), **facts))
        if not capabilities:
            raise SourceEditRefused("no_named_tensor_operands")
        return capabilities

    def _source_flow(self, op, expression, node):
        """Share the same source-order proof across structural edit recipes."""
        if expression.id not in self.source_flows:
            try:
                self.source_flows[expression.id] = StraightLineFlow(self, expression, node)
            except FlowRefused as error:
                raise SourceEditRefused(str(error)) from None
        flow = self.source_flows[expression.id]
        if (op.invocationId not in flow.states or flow.events[op.invocationId].id != op.id
                or {event.id for event in self.by_expression[expression.id]} != {event.id for event in flow.events.values()}):
            raise SourceEditRefused("ambiguous_source_scope")
        return flow

    def plan_reconnection(self, payload):
        intent = ReconnectArgumentEdit.model_validate(payload)
        capability = next((cap for cap in self.reconnect_capabilities(intent.target.semanticId)
                           if cap.argumentPath == intent.argumentPath), None)
        if (capability is None or intent.capabilityId != capability.id or intent.target != capability.target
                or intent.expected != capability.expected or intent.scope != capability.scope.kind):
            raise SourceEditRefused("stale_capability")
        choice = next((choice for choice in capability.choices if choice.value == intent.value), None)
        if choice is None:
            raise SourceEditRefused("unavailable_source_value")
        return self._patch(capability, None if intent.value == intent.expected else choice.name)

    def insert_unary_capabilities(self, operation_id):
        operands = self.reconnect_capabilities(operation_id)
        anchor = operands[0].expression.anchors[0]
        # New callees have no prior observation. Reject source-visible namespace
        # mutation/reflection rather than asserting a fresh alias avoids it.
        for node in ast.walk(self.indexes[anchor.file].tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("exec", "eval", "globals", "locals", "vars", "setattr", "delattr")
                    or isinstance(node, ast.Attribute) and (node.attr == "__dict__" or isinstance(node.ctx, ast.Del))):
                raise SourceEditRefused("unproven_unary_namespace")
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and _dotted_name(node):
                # Ordinary instance assignments remain supported; imported or
                # arbitrary module/namespace writes cannot justify a new callee.
                if not isinstance(node.value, ast.Name) or node.value.id != "self":
                    raise SourceEditRefused("unproven_unary_namespace")
        namespace, import_plan = self._torch_namespace(anchor)
        if len(namespace) > 256 or not namespace.isascii():
            raise SourceEditRefused("source_limit")
        caps = []
        for operand in operands:
            facts = operand.model_dump(mode="json", exclude={"id", "choices"})
            facts.update(command="insert_unary_op", recipe="straight_line_unary_insert_v1",
                namespace=namespace, importPlan=import_plan)
            caps.append(InsertUnaryCapability(id="edit:" + _identity(facts), **facts))
        return caps

    def plan_insert_unary(self, payload):
        intent = InsertUnaryEdit.model_validate(payload)
        cap = next((cap for cap in self.insert_unary_capabilities(intent.target.semanticId)
                    if cap.argumentPath == intent.argumentPath), None)
        if (cap is None or intent.capabilityId != cap.id or intent.target != cap.target
                or intent.expected != cap.expected or intent.scope != cap.scope.kind):
            raise SourceEditRefused("stale_capability")
        return self._patch(cap, cap.namespace + "." + intent.value + "(" + cap.expectedName + ")")

    def delete_unary_capability(self, operation_id):
        activation = self.activation_capability(operation_id)
        expression = activation.expression
        node = self._expression(expression)
        op = self.operations[operation_id]
        # One complete single-line call is replaced; comments/newlines outside
        # that call and every assignment/other use remain byte-for-byte intact.
        input_node = node.args[0] if node.args else next(kw.value for kw in node.keywords if kw.arg == "input")
        if (not isinstance(input_node, ast.Name) or len(input_node.id) > 128 or node.lineno != node.end_lineno
                or activation.options.inplace or op.output == op.arguments.args[0]):
            raise SourceEditRefused("unsupported_unary_deletion")
        flow = self._source_flow(op, expression, node)
        selected = flow.states[op.invocationId].get(input_node.id)
        if selected is None or selected.value != op.arguments.args[0]:
            raise SourceEditRefused("ambiguous_source_flow")
        for id, state in flow.states.items():
            item = state.get(input_node.id)
            if (item is None or item.definition is not selected.definition
                    or item.value != flow.events[id].arguments.args[0]):
                raise SourceEditRefused("ambiguous_source_flow")
        definition = flow.binding(selected.definition)
        if definition is None:
            raise SourceEditRefused("source_flow_limit")
        facts = dict(version=1, command="delete_unary_op", recipe="straight_line_unary_delete_v1",
            target=activation.target.model_dump(mode="json"), expression=expression.model_dump(mode="json"),
            binding=expression.model_dump(mode="json"), argumentPath=["args", 0], expected=op.output.model_dump(mode="json"),
            activation=activation.expected, input=dict(name=input_node.id, value=selected.value.model_dump(mode="json"), binding=definition),
            scope=activation.scope.model_dump(mode="json"))
        return DeleteUnaryCapability(id="edit:" + _identity(facts), **facts)

    def plan_delete_unary(self, payload):
        intent = DeleteUnaryEdit.model_validate(payload)
        cap = self.delete_unary_capability(intent.target.semanticId)
        if (intent.capabilityId != cap.id or intent.target != cap.target or intent.expected != cap.expected
                or intent.value != cap.input.value or intent.argumentPath != cap.argumentPath or intent.scope != cap.scope.kind):
            raise SourceEditRefused("stale_capability")
        return self._patch(cap, cap.input.name)

    def literal_capability(self, operation_id):
        op = self.operations.get(operation_id)
        if op is None:
            raise SourceEditRefused("unsupported_target")
        if len(op.sourceBindingIds) != 1:
            raise SourceEditRefused("ambiguous_binding")
        expression = self.bindings[op.sourceBindingIds[0]]
        node = self._expression(expression)
        if not isinstance(node, ast.BinOp) or type(node.op) not in _BINARY_TARGETS:
            raise SourceEditRefused("unsupported_expression")
        # Multi-line signed literals may contain comments. They need a separate
        # CST recipe; never erase those comments with a span replacement.
        expected = _number(node.right)
        if expected is None or node.right.lineno != node.right.end_lineno:
            raise SourceEditRefused("unsupported_expression")
        related = self.by_expression[expression.id]
        for observed in related:
            if observed.status != "completed" or observed.effects:
                raise SourceEditRefused("incomplete_or_mutating_observation")
            scalar = observed_binary_scalar(observed, _BINARY_TARGETS[type(node.op)])
            if scalar is None or scalar.model_dump(mode="json") != expected.model_dump(mode="json"):
                raise SourceEditRefused("derived_operation")
        anchor = expression.anchors[0]
        binding = source_binding(self.capture.source.id, self.files[anchor.file],
            self.sources[anchor.file].split("\n"), node.right, "argument", anchor.scope)
        if binding is None:
            raise SourceEditRefused("source_limit")
        calls = {event.invocationId for event in related if event.invocationId is not None}
        instances = {self.calls[id].moduleId for id in calls}
        facts = dict(version=1, command="set_literal", recipe="binary_scalar_v1",
            target=CaptureReference(version=1, captureId=self.capture.id, sourceId=self.capture.source.id,
                kind="operation", semanticId=op.id).model_dump(mode="json"),
            expression=expression.model_dump(mode="json"), binding=binding, argumentPath=["args", 1],
            expected=expected.model_dump(mode="json"), scope=dict(kind="source_expression", affects="all_executions",
                observedOperations=len(related), observedInvocations=len(calls), observedModuleInstances=len(instances),
                includesUnobservedExecutions=True))
        return LiteralEditCapability(id="edit:" + _identity(facts), **facts)

    def activation_capability(self, operation_id):
        op = self.operations.get(operation_id)
        if op is None:
            raise SourceEditRefused("unsupported_target")
        if len(op.sourceBindingIds) != 1:
            raise SourceEditRefused("ambiguous_binding")
        expression = self.bindings[op.sourceBindingIds[0]]
        node = self._expression(expression)
        if (not isinstance(node, ast.Call) or not isinstance(node.func, (ast.Attribute, ast.Name))
                or isinstance(node.func, ast.Attribute) and node.func.attr not in _ACTIVATION_TARGETS
                or node.func.lineno != node.func.end_lineno):
            raise SourceEditRefused("unsupported_activation_expression")
        anchor = expression.anchors[0]
        expected, namespace, import_plan = self._activation_name(node.func, anchor)
        supported = source_options(node, expected)
        if supported is None:
            raise SourceEditRefused("unsupported_activation_expression")
        options, compatible = supported
        if options.inplace and self.capture.coverage.omissions:
            raise SourceEditRefused("incomplete_or_mutating_observation")
        related = self.by_expression[expression.id]
        for observed in related:
            if observed.status != "completed":
                raise SourceEditRefused("incomplete_or_mutating_observation")
            if (len(observed.sourceBindingIds) != 1
                    or not observed_activation(observed, expected, options, self.values)):
                raise SourceEditRefused("derived_operation")
        binding = source_binding(self.capture.source.id, self.files[anchor.file],
            self.sources[anchor.file].split("\n"), node.func, "call", anchor.scope)
        if binding is None or len(namespace) > 256 or not namespace.isascii():
            raise SourceEditRefused("source_limit")
        calls = {event.invocationId for event in related if event.invocationId is not None}
        instances = {self.calls[id].moduleId for id in calls}
        facts = dict(version=1, command="replace_activation", recipe="functional_activation_v1",
            target=CaptureReference(version=1, captureId=self.capture.id, sourceId=self.capture.source.id,
                kind="operation", semanticId=op.id).model_dump(mode="json"),
            expression=expression.model_dump(mode="json"), binding=binding, namespace=namespace, importPlan=import_plan, expected=expected,
            options=options.model_dump(mode="json"), compatibleActivations=compatible,
            scope=dict(kind="source_expression", affects="all_executions", observedOperations=len(related),
                observedInvocations=len(calls), observedModuleInstances=len(instances), includesUnobservedExecutions=True))
        return ActivationEditCapability(id="edit:" + _identity(facts), **facts)

    def module_activation_capability(self, invocation_id):
        call = self.calls.get(invocation_id)
        if call is None:
            raise SourceEditRefused("unsupported_target")
        module = self.modules[call.moduleId]
        if len(module.constructorBindingIds) != 1:
            raise SourceEditRefused("ambiguous_constructor")
        expression = self.bindings[module.constructorBindingIds[0]]
        node = self._expression(expression)
        if (not isinstance(node, ast.Call) or not isinstance(node.func, (ast.Attribute, ast.Name))
                or isinstance(node.func, ast.Attribute) and node.func.attr not in _ACTIVATION_CLASSES.values()
                or node.func.lineno != node.func.end_lineno):
            raise SourceEditRefused("unsupported_activation_constructor")
        anchor = expression.anchors[0]
        expected, namespace, import_plan = self._activation_name(node.func, anchor, module=True)
        supported = source_options(node, expected, module=True)
        if supported is None:
            raise SourceEditRefused("unsupported_activation_constructor")
        options, compatible = supported
        class_name = _ACTIVATION_CLASSES[expected]
        related = self.by_constructor[expression.id]
        observed_calls = []
        for instance in related:
            if (len(instance.constructorBindingIds) != 1 or self.definitions[instance.definitionId].qualifiedName
                    != "torch.nn.modules.activation." + class_name):
                raise SourceEditRefused("derived_operation")
            observed_calls.extend(self.by_module[instance.id])
        # Missing retained events cannot certify one known activation per call.
        if self.capture.coverage.omissions:
            raise SourceEditRefused("incomplete_or_mutating_observation")
        for observed in observed_calls:
            ops = self.by_invocation[observed.id]
            if not observed_activation_call(observed, ops, self.child_calls[observed.id], expected, options, self.values):
                raise SourceEditRefused("derived_operation")
        binding = source_binding(self.capture.source.id, self.files[anchor.file],
            self.sources[anchor.file].split("\n"), node.func, "call", anchor.scope)
        if binding is None or len(namespace) > 256 or not namespace.isascii():
            raise SourceEditRefused("source_limit")
        facts = dict(version=1, command="replace_activation", recipe="module_activation_v1",
            target=CaptureReference(version=1, captureId=self.capture.id, sourceId=self.capture.source.id,
                kind="invocation", semanticId=call.id).model_dump(mode="json"),
            expression=expression.model_dump(mode="json"), binding=binding, namespace=namespace, importPlan=import_plan, expected=expected,
            options=options.model_dump(mode="json"), compatibleActivations=compatible,
            scope=dict(kind="source_expression", affects="all_executions", observedOperations=len(observed_calls),
                observedInvocations=len(observed_calls), observedModuleInstances=len(related), includesUnobservedExecutions=True))
        return ActivationEditCapability(id="edit:" + _identity(facts), **facts)

    def plan_activation(self, payload):
        intent = ReplaceActivationEdit.model_validate(payload)
        capability = (self.module_activation_capability(intent.target.semanticId) if intent.target.kind == "invocation"
                      else self.activation_capability(intent.target.semanticId))
        if (intent.capabilityId != capability.id or intent.target != capability.target
                or intent.expected != capability.expected or intent.scope != capability.scope.kind):
            raise SourceEditRefused("stale_capability")
        if intent.value not in capability.compatibleActivations:
            raise SourceEditRefused("incompatible_activation_options")
        name = _ACTIVATION_CLASSES[intent.value] if capability.recipe == "module_activation_v1" else intent.value
        replacement = None if intent.value == intent.expected else capability.namespace + "." + name
        return self._patch(capability, replacement)

    def _constructor_context(self, invocation_id):
        call = self.calls.get(invocation_id)
        if call is None:
            raise SourceEditRefused("unsupported_target")
        module = self.modules[call.moduleId]
        if len(module.constructorBindingIds) != 1:
            raise SourceEditRefused("ambiguous_constructor")
        expression = self.bindings[module.constructorBindingIds[0]]
        node = self._expression(expression)
        if (not isinstance(node, ast.Call) or any(isinstance(arg, ast.Starred) for arg in node.args)
                or any(kw.arg is None for kw in node.keywords)):
            raise SourceEditRefused("ambiguous_constructor_arguments")
        anchor = expression.anchors[0]
        qualified = self.definitions[module.definitionId].qualifiedName
        related = self.by_constructor[expression.id]
        if any(len(instance.constructorBindingIds) != 1 or instance.definitionId != module.definitionId for instance in related):
            raise SourceEditRefused("ambiguous_constructor")
        name = qualified.rsplit(".", 1)[-1]
        if not name.isidentifier() or len(name) > 128:
            raise SourceEditRefused("unproven_constructor")
        if not all(instance.constructorClassVerified for instance in related):
            # Older captures may still prove a built-in Torch constructor from
            # its import and qualified class. Custom/dynamic names require the
            # stronger observed direct-class identity for every shared instance.
            if anchor.file not in self.namespaces:
                self.namespaces[anchor.file] = _torch_imports(self.indexes[anchor.file].tree)
            spelling = _dotted_name(node.func)
            root, _, tail = (spelling or "").partition(".")
            imported = self.namespaces[anchor.file].get(root)
            resolved = imported + ("." + tail if tail else "") if imported else ""
            if resolved != "torch.nn." + name or not qualified.startswith("torch.nn.modules."):
                raise SourceEditRefused("unproven_constructor")
        candidates = [(["args", index], arg) for index, arg in enumerate(node.args)] + [
            (["kwargs", kw.arg], kw.value) for kw in node.keywords]
        if len(candidates) > 256:
            raise SourceEditRefused("source_limit")
        return call, expression, related, name, candidates

    def _constructor_counts(self, related):
        observed = {item.id for instance in related for item in self.by_module[instance.id]}
        descendants = set(observed)
        for invocation in self.capture.invocations:
            if invocation.parentId in descendants:
                descendants.add(invocation.id)
        return dict(observedOperations=sum(op.invocationId in descendants for op in self.capture.operations),
            observedInvocations=len(observed), observedModuleInstances=len(related))

    def constructor_literal_capabilities(self, invocation_id):
        call, expression, related, name, candidates = self._constructor_context(invocation_id)
        anchor = expression.anchors[0]
        scope = dict(kind="source_expression", affects="all_executions", includesUnobservedExecutions=True,
                     **self._constructor_counts(related))
        capabilities = []
        for path, argument in candidates:
            expected = _number(argument)
            if expected is None or argument.lineno != argument.end_lineno:
                continue  # Never inline a variable/config/expression using a measured value.
            binding = source_binding(self.capture.source.id, self.files[anchor.file],
                self.sources[anchor.file].split("\n"), argument, "argument", anchor.scope)
            if binding is None:
                continue
            facts = dict(version=1, command="set_constructor_literal", recipe="constructor_scalar_v1",
                target=CaptureReference(version=1, captureId=self.capture.id, sourceId=self.capture.source.id,
                    kind="invocation", semanticId=call.id).model_dump(mode="json"),
                expression=expression.model_dump(mode="json"), binding=binding, sourceArgumentPath=path,
                constructor=name, expected=expected.model_dump(mode="json"), scope=scope)
            capabilities.append(ConstructorLiteralCapability(id="edit:" + _identity(facts), **facts))
        if not capabilities:
            raise SourceEditRefused("no_constructor_literals")
        return capabilities

    def _configuration_reference(self, expression, reference):
        if not isinstance(reference, ast.Name) or len(reference.id) > 128:
            return None
        anchor = expression.anchors[0]
        if anchor.file not in self.configuration_indexes:
            self.configuration_indexes[anchor.file] = ConfigurationIndex(self.indexes[anchor.file].tree)
        found = self.configuration_indexes[anchor.file].resolve(reference, anchor.scope)
        if found is None:
            return None
        value, scope, count = found
        expected = _number(value)
        if expected is None or value.lineno != value.end_lineno:
            return None
        binding = source_binding(self.capture.source.id, self.files[anchor.file],
            self.sources[anchor.file].split("\n"), value, "configuration", scope)
        return (binding, expected, count) if binding is not None else None

    def _configuration_scope(self, binding, reference_count):
        if binding["id"] in self.configuration_scopes:
            return self.configuration_scopes[binding["id"]]
        related, sites = {}, set()
        for expression_id, instances in self.by_constructor.items():
            expression = self.bindings[expression_id]
            if expression.anchors[0].file != binding["anchors"][0]["file"]:
                continue
            try:
                node = self._expression(expression)
            except SourceEditRefused:
                continue  # Counts cover proven references; other uses are explicit.
            for reference in ast.walk(node):
                found = self._configuration_reference(expression, reference)
                if found is not None and found[0]["id"] == binding["id"]:
                    sites.add(expression_id)
                    related.update((instance.id, instance) for instance in instances)
                    break
        scope = dict(kind="configuration_definition", affects="all_references", knownSourceReferences=reference_count,
            observedConstructorSites=len(sites), includesUnobservedReferences=True,
            **self._constructor_counts(related.values()))
        self.configuration_scopes[binding["id"]] = scope
        return scope

    def configuration_literal_capabilities(self, invocation_id):
        call, expression, _related, _name, candidates = self._constructor_context(invocation_id)
        anchor = expression.anchors[0]
        capabilities = []
        for path, argument in candidates:
            found = self._configuration_reference(expression, argument)
            if found is None:
                continue
            binding, expected, count = found
            reference = source_binding(self.capture.source.id, self.files[anchor.file],
                self.sources[anchor.file].split("\n"), argument, "argument", anchor.scope)
            if reference is None:
                continue
            facts = dict(version=1, command="set_configuration_literal", recipe="configuration_scalar_v1",
                target=CaptureReference(version=1, captureId=self.capture.id, sourceId=self.capture.source.id,
                    kind="invocation", semanticId=call.id).model_dump(mode="json"), expression=expression.model_dump(mode="json"),
                reference=reference, binding=binding, sourceArgumentPath=path, symbol=argument.id,
                expected=expected.model_dump(mode="json"), scope=self._configuration_scope(binding, count))
            capabilities.append(ConfigurationLiteralCapability(id="edit:" + _identity(facts), **facts))
        if not capabilities:
            raise SourceEditRefused("no_configuration_literals")
        return capabilities

    def plan_configuration_literal(self, payload):
        intent = SetConfigurationLiteralEdit.model_validate(payload)
        capability = next((cap for cap in self.configuration_literal_capabilities(intent.target.semanticId)
                           if cap.sourceArgumentPath == intent.sourceArgumentPath), None)
        if (capability is None or intent.capabilityId != capability.id or intent.target != capability.target
                or intent.expected != capability.expected or intent.scope != capability.scope.kind):
            raise SourceEditRefused("stale_capability")
        if intent.value == intent.expected:
            return self._patch(capability, None)
        value = intent.value.value
        replacement = value if intent.value.kind == "int" else "-0.0" if value == "-0" else repr(float(value))
        return self._patch(capability, replacement)

    def plan_constructor_literal(self, payload):
        intent = SetConstructorLiteralEdit.model_validate(payload)
        capability = next((cap for cap in self.constructor_literal_capabilities(intent.target.semanticId)
                           if cap.sourceArgumentPath == intent.sourceArgumentPath), None)
        if (capability is None or intent.capabilityId != capability.id or intent.target != capability.target
                or intent.expected != capability.expected or intent.scope != capability.scope.kind):
            raise SourceEditRefused("stale_capability")
        if intent.value == intent.expected:
            return self._patch(capability, None)
        value = intent.value.value
        replacement = value if intent.value.kind == "int" else "-0.0" if value == "-0" else repr(float(value))
        return self._patch(capability, replacement)

    def plan_literal(self, payload):
        """Re-derive authority, then replace exactly one proven literal range.

        A capability supplied by a client is never accepted as authority. No-op
        requests preserve even numeric spelling and return no patches.
        """
        intent = SetLiteralEdit.model_validate(payload)
        capability = self.literal_capability(intent.target.semanticId)
        if (intent.capabilityId != capability.id or intent.target != capability.target
                or intent.argumentPath != capability.argumentPath or intent.expected != capability.expected
                or intent.scope != capability.scope.kind):
            raise SourceEditRefused("stale_capability")
        if intent.value == intent.expected:
            return self._patch(capability, None)
        value = intent.value.value
        replacement = value if intent.value.kind == "int" else "-0.0" if value == "-0" else repr(float(value))
        return self._patch(capability, replacement)

    def _patch(self, capability, replacement):
        anchor = capability.binding.anchors[0]
        before = self.sources[anchor.file]
        result = dict(file=anchor.file, expected_digest=anchor.digest, source_digest=anchor.digest,
            edits=[], capability=capability.model_dump(mode="json"), validation="static_only")
        if replacement is None:
            return result
        lines = before.split("\n")
        line = lines[anchor.startLine - 1]
        # These offsets came from the exact verified AST. Decode strictly so a
        # forged surrogate boundary cannot accidentally widen a replacement.
        start = len(line.encode("utf-16-le")[:anchor.startColumn * 2].decode("utf-16-le"))
        end = len(line.encode("utf-16-le")[:anchor.endColumn * 2].decode("utf-16-le"))
        lines[anchor.startLine - 1] = line[:start] + replacement + line[end:]
        edits = [dict(range=dict(start=[anchor.startLine, anchor.startColumn], end=[anchor.endLine, anchor.endColumn]), newText=replacement)]
        if isinstance(capability, (ActivationEditCapability, InsertUnaryCapability)) and capability.importPlan is not None:
            plan = capability.importPlan
            statement = f"import {plan.module} as {capability.namespace}{plan.lineEnding}"
            lines.insert(plan.line - 1, statement[:-1])
            edits.append(dict(range=dict(start=[plan.line, 0], end=[plan.line, 0]), newText=statement))
        after = "\n".join(lines)
        try:
            compile(after, anchor.file, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, RecursionError):
            raise SourceEditRefused("invalid_candidate") from None
        result.update(source_digest=hashlib.sha256(after.encode()).hexdigest(), edits=edits)
        return result

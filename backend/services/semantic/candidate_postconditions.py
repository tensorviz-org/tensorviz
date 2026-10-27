"""Bounded requested-change evidence over the existing recorder's saved capture.

Correspondence follows an exact reviewed source range, never operation IDs,
sequence numbers, tensor values or cross-run output equality. No code is executed.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass

from .contracts import CandidatePostconditions, CaptureSnapshot, SourceBinding, ActivationSourceOptions, ValueRef
from .construction import constructor_binding
from .execution_plan import applicability_digest
from .source_bindings import SourceIndex, source_binding
from .source_edits import observed_binary_scalar, CapturedEditResolver, _number
from .initializer_arguments import source_argument
from .candidate_configuration import ConfigurationPostcondition, prepare_configuration, check_configuration
from .source_flow import BINARY_TARGETS, StraightLineFlow
from .source_activations import NAMES, CLASSES, observed_activation, observed_activation_call, source_options


@dataclass(frozen=True)
class LiteralPostcondition:
    expression: SourceBinding | None
    target: str | None
    expected: dict
    recipe = "binary_scalar_v1"


@dataclass(frozen=True)
class ActivationPostcondition:
    expression: SourceBinding | None
    expected: str
    options: ActivationSourceOptions
    recipe: str


@dataclass(frozen=True)
class OperandPostcondition:
    expression: SourceBinding | None
    argument: int
    name: str
    source: str
    recipe: str


@dataclass(frozen=True)
class DeletionPostcondition:
    expression: SourceBinding | None
    replacement: SourceBinding | None
    name: str
    source: str
    recipe = "straight_line_unary_delete_v1"


@dataclass(frozen=True)
class ConstructorPostcondition:
    expression: SourceBinding | None
    argument: tuple
    expected: dict
    recipe = "constructor_scalar_v1"


def _mapped_deletion(snapshot, file_path, patch, reviewed_source):
    """Map the surviving name and its containing forward definition positively."""
    capability = patch["capability"]
    anchor = capability["expression"]["anchors"][0]
    name = capability["input"]["name"]
    if len(patch["edits"]) != 1:
        return None
    edit = patch["edits"][0]
    if (edit["newText"] != name or anchor["startLine"] != anchor["endLine"]
            or edit["range"] != dict(start=[anchor["startLine"], anchor["startColumn"]],
                                     end=[anchor["endLine"], anchor["endColumn"]])):
        return None
    file = next(file for file in snapshot.files if file.path == file_path)
    index = SourceIndex(snapshot.id, file, reviewed_source.encode(), retain_nodes=True)
    definitions = [(binding, node) for binding, node in index.definition_nodes.values()
        if binding["anchors"][0]["scope"] == anchor["scope"]]
    if len(definitions) != 1:
        return None
    definition, function = definitions[0]
    matches, lines = [], reviewed_source.split("\n")
    for node in ast.walk(function):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load) or node.id != name:
            continue
        binding = source_binding(snapshot.id, file, lines, node, "argument", anchor["scope"])
        found = binding["anchors"][0] if binding else None
        if found and (found["startLine"], found["startColumn"], found["endLine"], found["endColumn"]) == (
                anchor["startLine"], anchor["startColumn"], anchor["startLine"], anchor["startColumn"] + len(name.encode("utf-16-le")) // 2):
            matches.append(binding)
    if len(matches) != 1:
        return None
    return SourceBinding.model_validate(definition), SourceBinding.model_validate(matches[0])


def _mapped_expression(snapshot, file_path, patch, reviewed_source):
    """Map the resolver's contained single-line replacement and optional import."""
    anchor = patch["capability"]["expression"]["anchors"][0]
    replacements = [edit for edit in patch["edits"] if edit["range"]["start"] != edit["range"]["end"]]
    insertions = [edit for edit in patch["edits"] if edit["range"]["start"] == edit["range"]["end"]]
    if len(replacements) != 1 or len(insertions) > 1:
        return None
    edit = replacements[0]
    start, end = edit["range"]["start"], edit["range"]["end"]
    if (start[0] != end[0] or "\n" in edit["newText"]
            or start < [anchor["startLine"], anchor["startColumn"]]
            or end > [anchor["endLine"], anchor["endColumn"]]):
        return None
    finish = anchor["endColumn"]
    if end[0] == anchor["endLine"]:
        finish += len(edit["newText"].encode("utf-16-le")) // 2 - (end[1] - start[1])
    shift = 0
    if insertions:
        plan = patch["capability"].get("importPlan")
        if (plan is None or insertions[0]["range"]["start"] != [plan["line"], 0]
                or plan["line"] > anchor["startLine"]
                or insertions[0]["newText"] != f'import {plan["module"]} as {patch["capability"]["namespace"]}{plan["lineEnding"]}'):
            return None
        shift = 1
    file = next(file for file in snapshot.files if file.path == file_path)
    index = SourceIndex(snapshot.id, file, reviewed_source.encode("utf-8"), retain_nodes=True)
    matches = []
    for binding, node in index.bound_nodes.values():
        changed = binding["anchors"][0]
        if (changed["scope"], changed["startLine"], changed["startColumn"], changed["endLine"], changed["endColumn"]) == (
                anchor["scope"], anchor["startLine"] + shift, anchor["startColumn"], anchor["endLine"] + shift, finish):
            if patch["capability"]["expression"]["kind"] == "constructor":
                binding = constructor_binding(binding)
            matches.append((SourceBinding.model_validate(binding), node))
    return matches[0] if len(matches) == 1 else None


def prepare_postconditions(snapshot, file_path, patch, intent, reviewed_source):
    """Called after the resolver has proved the exact reviewed patch and bytes."""
    recipe = patch["capability"]["recipe"]
    if recipe == "configuration_scalar_v1":
        return prepare_configuration(snapshot, file_path, patch, intent, reviewed_source)
    if recipe == "binary_scalar_v1":
        unavailable = LiteralPostcondition(None, None, intent["value"])
    elif recipe in ("functional_activation_v1", "module_activation_v1"):
        unavailable = ActivationPostcondition(None, intent["value"],
            ActivationSourceOptions.model_validate(patch["capability"]["options"]), recipe)
    elif recipe in ("straight_line_tensor_v1", "straight_line_unary_insert_v1"):
        name = (next(item["name"] for item in patch["capability"]["choices"] if item["value"] == intent["value"])
                if recipe == "straight_line_tensor_v1" else patch["capability"]["expectedName"])
        unavailable = OperandPostcondition(None, intent["argumentPath"][1], name, reviewed_source, recipe)
    elif recipe == "straight_line_unary_delete_v1":
        unavailable = DeletionPostcondition(None, None, patch["capability"]["input"]["name"], reviewed_source)
    elif recipe == "constructor_scalar_v1":
        unavailable = ConstructorPostcondition(None, tuple(intent["sourceArgumentPath"]), intent["value"])
    else:
        return None
    try:
        if isinstance(unavailable, DeletionPostcondition):
            found = _mapped_deletion(snapshot, file_path, patch, reviewed_source)
            return DeletionPostcondition(*found, unavailable.name, reviewed_source) if found else unavailable
        found = _mapped_expression(snapshot, file_path, patch, reviewed_source)
        if found is None:
            return unavailable
        binding, node = found
        if isinstance(unavailable, ConstructorPostcondition):
            if (not isinstance(node, ast.Call) or len(patch["edits"]) != 1
                    or any(isinstance(arg, ast.Starred) for arg in node.args)
                    or any(keyword.arg is None for keyword in node.keywords)):
                return unavailable
            group, slot = unavailable.argument
            candidates = (node.args[slot:slot + 1] if group == "args" else
                [keyword.value for keyword in node.keywords if keyword.arg == slot])
            number = _number(candidates[0]) if len(candidates) == 1 else None
            if number is None or number.model_dump(mode="json") != unavailable.expected:
                return unavailable
            return ConstructorPostcondition(binding, unavailable.argument, unavailable.expected)
        if isinstance(unavailable, OperandPostcondition):
            if not isinstance(node, ast.BinOp) or type(node.op) not in BINARY_TARGETS:
                return unavailable
            operand = (node.left, node.right)[unavailable.argument]
            if recipe == "straight_line_unary_insert_v1":
                if (not isinstance(operand, ast.Call) or not isinstance(operand.func, ast.Attribute)
                        or operand.func.attr != intent["value"] or len(operand.args) != 1 or operand.keywords):
                    return unavailable
                operand = operand.args[0]
            elif len(patch["edits"]) != 1:
                return unavailable
            if not isinstance(operand, ast.Name) or operand.id != unavailable.name:
                return unavailable
            return OperandPostcondition(binding, unavailable.argument, unavailable.name, reviewed_source, recipe)
        if isinstance(unavailable, ActivationPostcondition):
            if not isinstance(node, ast.Call):
                return unavailable
            options = source_options(node, intent["value"], module=recipe == "module_activation_v1")
            if options is None or options[0] != unavailable.options:
                return unavailable
            return ActivationPostcondition(binding, unavailable.expected, unavailable.options, recipe)
        if not isinstance(node, ast.BinOp) or type(node.op) not in BINARY_TARGETS or len(patch["edits"]) != 1:
            return unavailable
        return LiteralPostcondition(binding, BINARY_TARGETS[type(node.op)], intent["value"])
    except (ValueError, KeyError, StopIteration, RecursionError, UnicodeError, SyntaxError):
        return unavailable


def _result(proof, execution_id, capture_id=None):
    fields = dict(version=1, executionId=execution_id, captureId=capture_id,
        recipe=proof.recipe if proof is not None else None,
        scope="observed_configuration_arguments" if isinstance(proof, ConfigurationPostcondition) else
            "observed_initializer_entries" if isinstance(proof, ConstructorPostcondition) else "observed_expression",
        status="unavailable", bindingId=None, observedOperations=0, observedInitializers=0)

    def result(reason, **updates):
        return CandidatePostconditions(**(fields | updates), reason=reason)
    return result


def check_postconditions(execution, proof, payload):
    """Pin the private candidate's source and plan before checking its observations."""
    result = _result(proof, execution.id)

    if proof is None:
        return result("unsupported_recipe")
    try:
        if isinstance(payload, CaptureSnapshot):
            payload = payload.model_dump(mode="json")
        capture = CaptureSnapshot.model_validate(payload)
    except (ValueError, TypeError):
        return result("capture_unavailable")
    if (capture.source != execution.plan.source or capture.attemptId != execution.id
            or capture.planDigest != applicability_digest(execution.plan)):
        return result("capture_identity_mismatch")
    return check_captured_postconditions(capture, proof)


def check_captured_postconditions(capture: CaptureSnapshot, proof):
    """Use a validated capture and proof mapped into its exact source snapshot.

    Candidate and applied-source callers independently establish provenance and
    source identity. This shared checker never certifies current editor state.
    """
    result = _result(proof, capture.attemptId, capture.id)
    if proof is None:
        return result("unsupported_recipe")
    if capture.status != "complete" or capture.backend != "runtime":
        return result("capture_incomplete")
    if proof.expression is None:
        return result("source_mapping_unavailable")
    location = proof.expression.anchors[0].model_dump(exclude={"tokenDigest"})
    bindings = [binding for binding in capture.sourceBindings if binding.kind == proof.expression.kind and any(
        anchor.model_dump(exclude={"tokenDigest"}) == location for anchor in binding.anchors)]
    if not bindings:
        return result("expression_not_observed")
    # A competing binding at this range cannot hide additional occurrences under
    # another ID. Runtime IDs alone never resolve source ambiguity.
    if len(bindings) != 1 or bindings[0] != proof.expression:
        return result("source_mapping_unavailable")
    binding = bindings[0]
    if isinstance(proof, ConfigurationPostcondition):
        return check_configuration(capture, binding, proof, result)
    if isinstance(proof, ConstructorPostcondition):
        related = [module for module in capture.modules if binding.id in module.constructorBindingIds]
        if not related:
            return result("expression_not_observed")
        if any(module.constructorBindingIds != [binding.id] or not module.constructorClassVerified for module in related):
            return result("ambiguous_observation")
        values = [source_argument(module.initializerArguments, proof.argument) for module in related]
        if any(value is None for value in values):
            return result("initializer_arguments_unavailable")
        matches = all(value.model_dump(mode="json") == proof.expected for value in values)
        return result("observed_requested_constructor" if matches else "observed_constructor_mismatch",
            status="verified" if matches else "failed", bindingId=binding.id, observedInitializers=len(related))
    if isinstance(proof, ActivationPostcondition):
        return _check_activation(capture, binding, proof, result)
    if isinstance(proof, OperandPostcondition):
        return _check_operand(capture, binding, proof, result)
    if isinstance(proof, DeletionPostcondition):
        return _check_deletion(capture, binding, proof, result)
    operations = [op for op in capture.operations if binding.id in op.sourceBindingIds]
    if not operations:
        return result("expression_not_observed")
    observations = [observed_binary_scalar(op, proof.target) for op in operations]
    if any(value is None for value in observations):
        return result("ambiguous_observation")
    matches = all(value.model_dump(mode="json") == proof.expected for value in observations)
    return result("observed_requested_literal" if matches else "observed_literal_mismatch",
        status="verified" if matches else "failed", bindingId=binding.id, observedOperations=len(operations))


def _check_deletion(capture, binding, proof, result):
    try:
        file = binding.anchors[0].file
        resolver = CapturedEditResolver(capture, {file: proof.source})
        index = SourceIndex(capture.source.id, resolver.files[file], proof.source.encode(), retain_nodes=True)
        resolver.indexes[file] = index
        resolver.remaining_sites -= index.site_count
        definition, function = index.definition_nodes[binding.id]
        if definition != binding.model_dump(mode="json") or proof.replacement is None:
            return result("source_mapping_unavailable")
        related = [item for item in capture.definitions if binding.id in item.sourceBindingIds]
        if any(item.sourceBindingIds != [binding.id] for item in related):
            return result("ambiguous_observation")
        definitions = {item.id for item in related}
        if not any(resolver.modules[call.moduleId].definitionId in definitions for call in capture.invocations):
            return result("expression_not_observed")
        lines, replacement = proof.source.split("\n"), proof.replacement.model_dump(mode="json")
        matches = [node for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            and node.id == proof.name and source_binding(capture.source.id, resolver.files[file], lines,
                node, "argument", binding.anchors[0].scope) == replacement]
        if len(matches) != 1:
            return result("source_mapping_unavailable")
        flow = StraightLineFlow(resolver, binding, matches[0])
        if any(proof.name not in state or not isinstance(state[proof.name].value, ValueRef) for state in flow.states.values()):
            return result("ambiguous_observation")
    except (ValueError, KeyError, StopIteration, RecursionError, UnicodeError, SyntaxError):
        return result("ambiguous_observation")
    return result("observed_requested_deletion", status="verified", bindingId=binding.id, observedOperations=len(flow.events))


def _check_operand(capture, binding, proof, result):
    operations = [op for op in capture.operations if binding.id in op.sourceBindingIds]
    if not operations:
        return result("expression_not_observed")
    try:
        resolver = CapturedEditResolver(capture, {binding.anchors[0].file: proof.source})
        node = resolver._expression(binding)
        flow = StraightLineFlow(resolver, binding, node, deferred_argument=proof.argument)
        if {op.id for op in operations} != {event.id for event in flow.events.values()}:
            return result("ambiguous_observation")
        inserted = None
        if proof.recipe == "straight_line_unary_insert_v1":
            unary_binding = flow.binding((node.left, node.right)[proof.argument], "call")
            unary_ops = resolver.by_expression[unary_binding["id"]] if unary_binding else []
            inserted = {op.invocationId: op for op in unary_ops}
            if len(inserted) != len(unary_ops) or set(inserted) != set(flow.events):
                return result("ambiguous_observation")
        matches = []
        for call, state in flow.states.items():
            expected = state.get(proof.name)
            actual = flow.events[call].arguments.args[proof.argument]
            if expected is None or not isinstance(expected.value, ValueRef) or not isinstance(actual, ValueRef):
                return result("ambiguous_observation")
            if inserted is not None:
                # The flow already proved its imported callee, nonmutating
                # kernel and full input. Its output must feed this exact use.
                unary = inserted[call]
                if unary.arguments.args != [expected.value]:
                    return result("ambiguous_observation")
                matches.append(actual == unary.output)
            else:
                matches.append(actual == expected.value)
    except (ValueError, KeyError, StopIteration, RecursionError, UnicodeError, SyntaxError):
        return result("ambiguous_observation")
    name = "insertion" if inserted is not None else "argument"
    return result(f"observed_requested_{name}" if all(matches) else f"observed_{name}_mismatch",
        status="verified" if all(matches) else "failed", bindingId=binding.id, observedOperations=len(matches))


def _check_activation(capture, binding, proof, result):
    values = {value.id: value for value in capture.values}
    observed = []
    if proof.recipe == "functional_activation_v1":
        operations = [op for op in capture.operations if binding.id in op.sourceBindingIds]
        for op in operations:
            if op.sourceBindingIds != [binding.id]:
                return result("ambiguous_observation")
            names = [name for name in NAMES if observed_activation(op, name, proof.options, values)]
            if len(names) != 1:
                return result("ambiguous_observation")
            observed.append(names[0])
    else:
        definitions = {definition.id: definition for definition in capture.definitions}
        by_module, by_invocation, children = defaultdict(list), defaultdict(list), defaultdict(list)
        for call in capture.invocations:
            by_module[call.moduleId].append(call)
            children[call.parentId].append(call)
        for op in capture.operations:
            by_invocation[op.invocationId].append(op)
        for module in capture.modules:
            if binding.id not in module.constructorBindingIds:
                continue
            if module.constructorBindingIds != [binding.id]:
                return result("ambiguous_observation")
            name = next((name for name, cls in CLASSES.items() if definitions[module.definitionId].qualifiedName
                         == "torch.nn.modules.activation." + cls), None)
            if name is None:
                return result("ambiguous_observation")
            for call in by_module[module.id]:
                if not observed_activation_call(call, by_invocation[call.id], children[call.id], name, proof.options, values):
                    return result("ambiguous_observation")
                observed.append(name)
    if not observed:
        return result("expression_not_observed")
    matches = all(name == proof.expected for name in observed)
    return result("observed_requested_activation" if matches else "observed_activation_mismatch",
        status="verified" if matches else "failed", bindingId=binding.id, observedOperations=len(observed))

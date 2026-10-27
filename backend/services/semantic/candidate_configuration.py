"""Exact configuration definition and observed constructor-argument witnesses.

Only directly bound names are checked. Computed references inside a recorded
constructor cannot silently inherit another argument's successful check.
"""
import ast
from dataclasses import dataclass
import json

from .contracts import SourceBinding
from .construction import constructor_binding
from .initializer_arguments import source_argument
from .source_bindings import SourceIndex, source_binding
from .source_configuration import ConfigurationIndex
from .source_edits import CapturedEditResolver, _number


@dataclass(frozen=True)
class ConfigurationPostcondition:
    expression: SourceBinding | None
    definition: SourceBinding | None
    argument: tuple
    expected: dict
    source: str
    recipe = "configuration_scalar_v1"


def prepare_configuration(snapshot, file_path, patch, intent, source):
    unavailable = ConfigurationPostcondition(None, None, tuple(intent["sourceArgumentPath"]), intent["value"], source)
    try:
        if len(patch["edits"]) != 1:
            return unavailable
        edit, capability = patch["edits"][0], patch["capability"]
        definition, call = capability["binding"]["anchors"][0], capability["expression"]["anchors"][0]
        start, end = edit["range"]["start"], edit["range"]["end"]
        if (start != [definition["startLine"], definition["startColumn"]]
                or end != [definition["endLine"], definition["endColumn"]]
                or start[0] != end[0] or "\n" in edit["newText"]):
            return unavailable
        delta = len(edit["newText"].encode("utf-16-le")) // 2 - (end[1] - start[1])

        def moved(line, column):
            if start <= [line, column] < end:
                raise ValueError("configuration edit overlaps constructor")
            return line, column + (delta if line == end[0] and column >= end[1] else 0)

        beginning, finish = moved(call["startLine"], call["startColumn"]), moved(call["endLine"], call["endColumn"])
        file = next(file for file in snapshot.files if file.path == file_path)
        index = SourceIndex(snapshot.id, file, source.encode(), retain_nodes=True)
        matches = [(binding, node) for binding, node in index.bound_nodes.values()
            if (binding["anchors"][0]["scope"], binding["anchors"][0]["startLine"], binding["anchors"][0]["startColumn"],
                binding["anchors"][0]["endLine"], binding["anchors"][0]["endColumn"]) == (call["scope"], *beginning, *finish)]
        if len(matches) != 1:
            return unavailable
        binding, node = matches[0]
        if (not isinstance(node, ast.Call) or any(isinstance(arg, ast.Starred) for arg in node.args)
                or any(keyword.arg is None for keyword in node.keywords)):
            return unavailable
        group, slot = unavailable.argument
        arguments = node.args[slot:slot + 1] if group == "args" else [keyword.value for keyword in node.keywords if keyword.arg == slot]
        if len(arguments) != 1 or not isinstance(arguments[0], ast.Name) or arguments[0].id != capability["symbol"]:
            return unavailable
        found = ConfigurationIndex(index.tree).resolve(arguments[0], call["scope"])
        if found is None:
            return unavailable
        value, scope, _count = found
        number = _number(value)
        mapped = source_binding(snapshot.id, file, source.split("\n"), value, "configuration", scope)
        anchor = mapped["anchors"][0] if mapped else None
        if (number is None or number.model_dump(mode="json") != intent["value"] or anchor is None
                or (anchor["scope"], anchor["startLine"], anchor["startColumn"], anchor["endLine"], anchor["endColumn"])
                != (definition["scope"], *start, end[0], end[1] + delta)):
            return unavailable
        return ConfigurationPostcondition(SourceBinding.model_validate(constructor_binding(binding)),
            SourceBinding.model_validate(mapped), unavailable.argument, unavailable.expected, source)
    except (ValueError, KeyError, StopIteration, RecursionError, UnicodeError, SyntaxError):
        return unavailable


def check_configuration(capture, selected, proof, result):
    if proof.definition is None:
        return result("source_mapping_unavailable")
    try:
        file = proof.definition.anchors[0].file
        resolver = CapturedEditResolver(capture, {file: proof.source})
        uses, instances, matches = [], set(), []
        for binding_id, modules in resolver.by_constructor.items():
            binding = resolver.bindings[binding_id]
            if binding.anchors[0].file != file:
                continue
            node = resolver._expression(binding)
            if not isinstance(node, ast.Call):
                return result("configuration_reference_unavailable")
            references = set()
            for reference in ast.walk(node):
                found = resolver._configuration_reference(binding, reference)
                if found is not None and found[0] == proof.definition.model_dump(mode="json"):
                    references.add(reference)
            if not references:
                continue
            arguments = [(["args", index], argument) for index, argument in enumerate(node.args)] + [
                (["kwargs", keyword.arg], keyword.value) for keyword in node.keywords]
            direct = [(path, argument) for path, argument in arguments if argument in references]
            if (set(argument for _path, argument in direct) != references
                    or any(isinstance(argument, ast.Starred) for argument in node.args)
                    or any(keyword.arg is None for keyword in node.keywords)):
                return result("configuration_reference_unavailable")
            location = binding.anchors[0].model_dump(exclude={"tokenDigest"})
            if len([other for other in capture.sourceBindings if other.kind == "constructor" and any(
                    anchor.model_dump(exclude={"tokenDigest"}) == location for anchor in other.anchors)]) != 1:
                return result("source_mapping_unavailable")
            if any(module.constructorBindingIds != [binding_id] or not module.constructorClassVerified for module in modules):
                return result("ambiguous_observation")
            for path, _argument in direct:
                uses.append(dict(bindingId=binding_id, sourceArgumentPath=path))
                if len(uses) > 256 or len(json.dumps(uses, ensure_ascii=False, separators=(",", ":")).encode()) > 32768:
                    return result("configuration_reference_unavailable")
                for module in modules:
                    value = source_argument(module.initializerArguments, path)
                    if value is None:
                        return result("initializer_arguments_unavailable")
                    instances.add(module.id)
                    matches.append(value.model_dump(mode="json") == proof.expected)
        if not any(use["bindingId"] == selected.id and tuple(use["sourceArgumentPath"]) == proof.argument for use in uses):
            return result("expression_not_observed")
        passed = all(matches)
        return result("observed_requested_configuration" if passed else "observed_configuration_mismatch",
            status="verified" if passed else "failed", bindingId=selected.id,
            observedInitializers=len(instances), configurationUses=uses)
    except (ValueError, KeyError, StopIteration, RecursionError, UnicodeError, SyntaxError):
        return result("configuration_reference_unavailable")

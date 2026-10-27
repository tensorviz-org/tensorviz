"""Small constructor examples from public operator contracts, never model names.

This is a bounded syntax interpreter. A proposal is still only an example until
the normal isolated worker captures the actual model with it.
"""
from __future__ import annotations

import ast
import copy
import operator
from dataclasses import dataclass

from .graph_setup import _MISSING
from .module_index import module_of_file


@dataclass(frozen=True)
class Dimension:
    name: str


@dataclass(frozen=True)
class Expression:
    operation: str
    left: object
    right: object


OPERATIONS = {'Add': operator.add, 'Sub': operator.sub, 'Mult': operator.mul,
              'FloorDiv': operator.floordiv, 'Mod': operator.mod}


def symbolic(left, operation, right):
    if (type(operation).__name__ in OPERATIONS
            and isinstance(left, (Dimension, Expression, int)) and type(left) is not bool
            and isinstance(right, (Dimension, Expression, int)) and type(right) is not bool
            and any(isinstance(value, (Dimension, Expression)) for value in (left, right))):
        return Expression(type(operation).__name__, left, right)
    return _MISSING


def names(value):
    if isinstance(value, Dimension):
        return {value.name}
    return names(value.left) | names(value.right) if isinstance(value, Expression) else set()


def number(value, assignment):
    if isinstance(value, Dimension):
        return assignment.get(value.name, _MISSING)
    if isinstance(value, Expression):
        left, right = number(value.left, assignment), number(value.right, assignment)
        if type(left) is not int or type(right) is not int or max(abs(left), abs(right)) > 1_000_000:
            return _MISSING
        try:
            return OPERATIONS[value.operation](left, right)
        except ZeroDivisionError:
            return _MISSING
    return value


# Only arguments whose public contract supplies an integer dimension domain.
# These preferred values are explicitly synthetic, never recovered author intent.
FIELDS = {
    'Linear': [('in_features', 8), ('out_features', 8)],
    'Embedding': [('num_embeddings', 32), ('embedding_dim', 8)],
    'MultiheadAttention': [('embed_dim', 8), ('num_heads', 2)],
    'TransformerEncoderLayer': [('d_model', 8), ('nhead', 2)],
    'TransformerDecoderLayer': [('d_model', 8), ('nhead', 2)],
    'Transformer': [('d_model', 8), ('nhead', 2)],
    'RNN': [('input_size', 8), ('hidden_size', 8)],
    'GRU': [('input_size', 8), ('hidden_size', 8)],
    'LSTM': [('input_size', 8), ('hidden_size', 8)],
}
for _rank in (1, 2, 3):
    for _kind in ('Conv', 'ConvTranspose'):
        FIELDS[f'{_kind}{_rank}d'] = [('in_channels', 8), ('out_channels', 8), ('kernel_size', 3)]


def propose(resolver, construction):
    from .graph_setup_resolution import Instance, Symbol
    candidate = copy.deepcopy(construction, {id(_MISSING): _MISSING})
    slots = {}
    constraints = []
    evidence = {}

    def seed(instance, fields, prefix):
        supplied = resolver.parameters(instance)
        positional = [f for f in fields if f['kind'] in ('positional', 'keyword')]
        for field in fields:
            if not field['required'] or supplied.get(field['name'], _MISSING) is not _MISSING:
                continue
            name = prefix + field['name']
            if field in positional and positional.index(field) < len(instance.args):
                slot = instance.args, positional.index(field)
            elif field['kind'] == 'positional':
                # Preserve positional gaps; an unrelated missing argument cannot
                # be silently replaced just to reach a later dimension.
                if positional.index(field) != len(instance.args):
                    continue
                instance.args.append(_MISSING)
                slot = instance.args, len(instance.args) - 1
            elif field['kind'] in ('keyword', 'keyword_only'):
                slot = instance.kwargs, field['name']
            else:
                continue
            slots[name] = slot
            slot[0][slot[1]] = Dimension(name)

    seed(candidate, resolver.discovery['init'] or [], 'init.')
    for value in [*candidate.args, *candidate.kwargs.values()]:
        if isinstance(value, Instance):
            config = next((c for c in resolver.discovery['configs']
                           if (c['file'], c['cls']) == (value.symbol.file, value.symbol.name)
                           and resolver.config_supported(c)), None)
            if config:
                seed(value, config['fields'] or [], 'config.')
    if not slots or len(slots) > 8:
        return construction, []

    def constrain(kind, values, preferred, file, line, detail):
        used = set().union(*(names(v) for v in values))
        if not used or any(v is _MISSING for v in values):
            return
        constraints.append((kind, values, used))
        if kind == 'positive':
            for name in used:
                evidence.setdefault(name, (preferred, file, line, detail))

    def visit(value, file, line, trail=()):
        if not resolver.available() or len(trail) >= 12:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item, file, line, trail)
            return
        if not isinstance(value, Instance):
            return
        if not value.symbol.file:
            kind = value.symbol.name.removeprefix('torch.nn.')
            if value.symbol.name == 'torch.nn.' + kind and kind in FIELDS:
                fields = FIELDS[kind]
                def get(index):
                    return value.kwargs.get(fields[index][0], value.args[index] if index < len(value.args) else _MISSING)
                for index, (field, preferred) in enumerate(fields):
                    constrain('positive', [get(index)], preferred, file, line, f'{kind}.{field}')
                if kind in ('MultiheadAttention', 'TransformerEncoderLayer', 'TransformerDecoderLayer', 'Transformer'):
                    constrain('divisible', [get(0), get(1)], 8, file, line, kind)
                if kind.startswith('Conv'):
                    groups = value.kwargs.get('groups', value.args[6] if len(value.args) > 6 else 1)
                    constrain('positive', [groups], 1, file, line, f'{kind}.groups')
                    constrain('divisible', [get(0), groups], 8, file, line, kind)
                    constrain('divisible', [get(1), groups], 8, file, line, kind)
            for item in [*value.args, *value.kwargs.values()]:
                visit(item, file, line, trail)
            return
        key = value.symbol.file, value.symbol.name
        if key in trail:
            return
        definition = resolver.reader.definition(*key)
        if definition is None or not resolver.reader.decorators_supported(key[0], definition):
            return
        function = next((n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
        if function is None:
            if len(definition.bases) == 1:
                parent = resolver.reader.resolve(key[0], definition.bases[0])
                if parent:
                    visit(Instance(Symbol(parent.file, parent.name), value.args, value.kwargs), file, line, (*trail, key))
            return
        if not resolver.reader.decorators_supported(key[0], function, definition):
            return
        env = resolver.parameters(value)
        attributes = {}
        env['self'] = attributes

        def count_example(call):
            if not isinstance(call, ast.Call) or len(call.args) != 1 or call.keywords:
                return
            function = resolver.value(key[0], call.func, env)
            builtin = isinstance(function, Symbol) and not function.file and function.name == 'builtins.range'
            if isinstance(call.func, ast.Name) and call.func.id == 'range' and 'range' not in env:
                record = resolver.reader.index.modules[module_of_file(key[0])]
                builtin = (builtin or ('range' not in record.bound_names and not record.exports_dynamic
                                      and not record.unresolved_wildcard))
            if builtin:
                constrain('positive', [resolver.value(key[0], call.args[0], env)], 1,
                          key[0], call.lineno, 'range count for a small nonempty example')

        for statement in function.body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
                nodes = list(ast.walk(statement.value)) if statement.value is not None else []
                comprehensions = [node for node in nodes if isinstance(node, (ast.ListComp, ast.GeneratorExp))]
                if len(comprehensions) == 1 and not any(isinstance(node, ast.Lambda) for node in nodes):
                    generators = comprehensions[0].generators
                    if len(generators) == 1 and not generators[0].ifs and not generators[0].is_async:
                        count_example(generators[0].iter)
                result = resolver.value(key[0], statement.value, env)
                visit(result, key[0], statement.lineno, (*trail, key))
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = result
                    elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        attributes[target.attr] = result
                    elif isinstance(target, ast.Attribute):
                        # A config write invalidates its aliases; do not use a
                        # stale constructor field after an unknown mutation.
                        changed = resolver.value(key[0], target.value, env)
                        if changed is not _MISSING:
                            for bindings in (env, attributes):
                                for name, bound in list(bindings.items()):
                                    if bound is changed:
                                        bindings[name] = _MISSING
            elif isinstance(statement, ast.Assert) and isinstance(statement.test, ast.Compare) and len(statement.test.ops) == 1:
                compare = statement.test
                kind = type(compare.ops[0]).__name__
                if kind in ('Eq', 'Lt', 'LtE', 'Gt', 'GtE'):
                    constrain(kind, [resolver.value(key[0], compare.left, env),
                              resolver.value(key[0], compare.comparators[0], env)], 8, key[0], statement.lineno, 'constructor assertion')
            elif isinstance(statement, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AugAssign, ast.Delete)):
                if isinstance(statement, ast.For):
                    count_example(statement.iter)
                # Do not use a stale binding after code we cannot interpret.
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                        env[node.id] = _MISSING
                    elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)) and isinstance(node.value, ast.Name) and node.value.id == 'self':
                        attributes[node.attr] = _MISSING

    visit(candidate, resolver.file, 1)
    chosen = {}
    ordered = sorted(evidence)
    comparisons = {'Eq': operator.eq, 'Lt': operator.lt, 'LtE': operator.le, 'Gt': operator.gt, 'GtE': operator.ge}

    def valid():
        for kind, values, used in constraints:
            if not used <= chosen.keys():
                continue
            numbers = [number(value, chosen) for value in values]
            if any(type(n) is not int for n in numbers):
                return False
            if kind == 'positive' and not 1 <= numbers[0] <= 1_000_000:
                return False
            if kind == 'divisible' and (numbers[1] <= 0 or numbers[0] % numbers[1]):
                return False
            if kind in comparisons and not comparisons[kind](*numbers):
                return False
        return True

    attempts = 0
    def search(index):
        nonlocal attempts
        if index == len(ordered):
            return valid()
        name = ordered[index]
        preferred = evidence[name][0]
        for value in sorted(range(1, 257), key=lambda n: (abs(n-preferred), n)):
            attempts += 1
            if attempts > 4096 or not resolver.available():
                return False
            chosen[name] = value
            if valid() and search(index+1):
                return True
        chosen.pop(name, None)
        return False

    if not ordered or not search(0):
        return construction, []
    notes = []
    for name, (container, key) in slots.items():
        if name.startswith('config.') and name not in chosen and isinstance(container, dict):
            container.pop(key)
        else:
            container[key] = chosen.get(name, _MISSING)
        if name in chosen:
            _, file, line, detail = evidence[name]
            notes.append((name, chosen[name], file, line, detail))
    return candidate, notes

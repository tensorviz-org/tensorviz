"""Bounded source-based example calls. Never import or execute user code."""
from __future__ import annotations

import ast
import copy
import math
import re
from dataclasses import dataclass

from ..ingest.input_declarations import applicable
from .graph_setup import _MISSING, _literal, argument


@dataclass
class Symbol:
    file: str
    name: str


@dataclass
class Instance:
    symbol: Symbol
    args: list
    kwargs: dict


def _preserves_shape(layer):
    return (isinstance(layer, Instance) and not layer.symbol.file and layer.symbol.name in {
        'torch.nn.Identity', 'torch.nn.LayerNorm', 'torch.nn.RMSNorm', 'torch.nn.Dropout',
        'torch.nn.ReLU', 'torch.nn.GELU', 'torch.nn.SiLU', 'torch.nn.Tanh', 'torch.nn.Sigmoid',
        'torch.nn.LeakyReLU', 'torch.nn.ELU', 'torch.nn.Softplus', 'torch.nn.Mish', 'torch.nn.Hardswish'})


def _plain(value):
    try:
        argument(value)
        return True
    except (ValueError, TypeError, RecursionError):
        return False


class Resolver:
    def __init__(self, reader, file, cls, discovery):
        self.reader, self.file, self.cls, self.discovery = reader, file, cls, discovery
        self.evidence = []
        self.steps = 0
        self.unsupported_construction = False

    def available(self):
        self.steps += 1
        return self.steps <= 20_000

    def note(self, field, kind, detail, file=None, line=None):
        entry = dict(field=field, kind=kind, detail=detail[:512])
        if file is not None:
            entry.update(file=file, line=line or 1)
        if entry not in self.evidence and len(self.evidence) < 64:
            self.evidence.append(entry)

    def value(self, file, node, env):
        if not self.available():
            return _MISSING
        literal = _literal(node)
        if literal is not _MISSING:
            return literal
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            target = self.reader.resolve(file, node)
            if target:
                return Symbol(target.file, target.name)
            if any(item['file'] == file and item['cls'] == node.id and self.config_supported(item)
                   for item in self.discovery['configs']):
                return Symbol(file, node.id)
        if isinstance(node, ast.Attribute):
            parent = self.value(file, node.value, env)
            if isinstance(parent, dict):
                return parent.get(node.attr, _MISSING)
            if isinstance(parent, Instance):
                return self.attributes(parent).get(node.attr, _MISSING)
            if isinstance(node.value, ast.Name) and node.value.id in env:
                return _MISSING
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [self.value(file, item, env) for item in node.elts]
            if all(value is not _MISSING for value in values):
                return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.BinOp):
            left, right = self.value(file, node.left, env), self.value(file, node.right, env)
            from .graph_setup_constraints import symbolic
            expression = symbolic(left, node.op, right)
            if expression is not _MISSING:
                return expression
            if type(left) is int and type(right) is int and abs(left) <= 1_000_000 and abs(right) <= 1_000_000:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.FloorDiv) and right:
                    return left // right
                if isinstance(node.op, ast.Mod) and right:
                    return left % right
        if isinstance(node, (ast.Name, ast.Attribute)):
            symbol = self.reader.torch_symbol(file, node)
            if symbol:
                return Symbol('', symbol)
        if isinstance(node, ast.Call):
            function = self.value(file, node.func, env)
            args = [self.value(file, item, env) for item in node.args]
            kwargs = {item.arg: self.value(file, item.value, env) for item in node.keywords}
            if isinstance(function, Symbol) and None not in kwargs:
                return Instance(function, args, kwargs)
        return _MISSING

    def config_supported(self, option):
        definition = self.reader.definition(option['file'], option['cls'])
        if (definition is None or definition.keywords or definition.bases or not all(
                self.reader.torch_symbol(option['file'], node.func if isinstance(node, ast.Call) else node) == 'dataclasses.dataclass'
                for node in definition.decorator_list)):
            return False
        methods = [node for node in definition.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if not methods:
            return True
        if len(methods) != 1 or not isinstance(methods[0], ast.FunctionDef):
            return False
        constructor = methods[0]
        positional = constructor.args.posonlyargs + constructor.args.args
        if (constructor.name != '__init__' or constructor.decorator_list or not positional
                or constructor.args.vararg or constructor.args.kwarg):
            return False
        owner = positional[0].arg
        for statement in constructor.body:
            if isinstance(statement, ast.Pass) or (isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)):
                continue
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                return False
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if not all(isinstance(target, ast.Name) and target.id != owner
                       or isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                       and target.value.id == owner for target in targets):
                return False
            if any(isinstance(node, (ast.Call, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom))
                   for node in ast.walk(statement)):
                return False
        return True

    def attributes(self, instance):
        """A config constructor parameter is not necessarily a stored attribute."""
        option = next((item for item in self.discovery['configs']
                       if (item['file'], item['cls']) == (instance.symbol.file, instance.symbol.name)), None)
        if not option:
            return self.parameters(instance)
        if not self.config_supported(option):
            return {}
        definition = self.reader.definition(option['file'], option['cls'])
        constructor = next((node for node in definition.body if isinstance(node, ast.FunctionDef)
                            and node.name == '__init__'), None)
        if constructor is None:
            return self.parameters(instance)
        attributes = {}
        for statement in definition.body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value is not None:
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        attributes[target.id] = self.value(option['file'], statement.value, {})
        owner = (constructor.args.posonlyargs + constructor.args.args)[0].arg
        env = {**self.parameters(instance), owner: attributes}
        for statement in constructor.body:
            if not self.available():
                return {}
            if isinstance(statement, (ast.Assign, ast.AnnAssign)) and statement.value is not None:
                value = self.value(option['file'], statement.value, env)
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value
                    else:
                        attributes[target.attr] = value
        return attributes

    def parameters(self, instance):
        fields = self.reader.method(instance.symbol.file, instance.symbol.name, '__init__')
        config = next((item for item in self.discovery['configs']
                       if (item['file'], item['cls']) == (instance.symbol.file, instance.symbol.name)), None)
        if config:
            fields = config['fields']
        if fields is None:
            return {}
        result = {field['name']: field.get('defaultValue', _MISSING) for field in fields}
        definition = self.reader.definition(instance.symbol.file, instance.symbol.name)
        function = next((node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == '__init__'), None) if definition else None
        if function:
            positional = function.args.posonlyargs + function.args.args
            for parameter, default in zip(positional[-len(function.args.defaults):], function.args.defaults):
                if result.get(parameter.arg) is _MISSING:
                    result[parameter.arg] = self.value(instance.symbol.file, default, {})
        positional = [field for field in fields if field['kind'] in ('positional', 'keyword')]
        supplied = {field['name']: value for field, value in zip(positional, instance.args)}
        supplied.update(instance.kwargs)
        for name, value in supplied.items():
            field = next((f for f in fields if f['name'] == name), {})
            if value is not _MISSING or 'defaultValue' not in field:
                result[name] = value
        return result

    def constructor(self, instance):
        """Convert source construction into the existing profile binding grammar."""
        fields = self.discovery['init']
        if fields is None:
            return None
        positional = [field for field in fields if field['kind'] in ('positional', 'keyword')]
        if len(instance.args) > len(positional):
            return None
        binding, selection, unknown, defaults = dict(args=[], kwargs={}), None, [], []
        supplied = [(positional[i], value) for i, value in enumerate(instance.args)]
        for name, value in instance.kwargs.items():
            field = next((f for f in fields if f['name'] == name), None)
            if field is None or field['kind'] == 'positional' or name in [f['name'] for f in positional[:len(instance.args)]]:
                return None
            supplied.append((field, value))
        for field, value in supplied:
            if value is _MISSING:
                if 'defaultValue' in field:
                    defaults.append(field['name'])
                else:
                    unknown.append(field['name'])
                continue
            if isinstance(value, Instance):
                options = [(i, item) for i, item in enumerate(self.discovery['configs'])
                           if (item['file'], item['cls']) == (value.symbol.file, value.symbol.name)]
                if len(options) != 1 or not self.config_supported(options[0][1]) or selection or field['kind'] == 'positional' or value.args:
                    return None
                if not all(_plain(v) for v in value.kwargs.values()):
                    return None
                selection = dict(index=options[0][0], parameter=field['name'], values=value.kwargs)
            elif _plain(value):
                if field['kind'] == 'positional':
                    if len(binding['args']) != positional.index(field):
                        unknown.append(field['name'])
                        continue
                    binding['args'].append(argument(value))
                else:
                    binding['kwargs'][field['name']] = argument(value)
            else:
                return None
        return binding, selection, unknown, defaults

    def source_constructions(self):
        result = []
        for file, tree in self.reader.trees.items():
            factory_names = {factory['callable'] for factory in self.discovery['factories'] if factory['file'] == file}
            factory_nodes = {node for definition in tree.body
                             if isinstance(definition, ast.FunctionDef) and definition.name in factory_names
                             for node in ast.walk(definition)}
            for owner in ast.walk(tree):
                # A builder's branches are alternatives, not selected repository
                # calls that can override known constructor arguments.
                if owner in factory_nodes:
                    continue
                body = getattr(owner, 'body', None)
                if not isinstance(body, list):
                    continue
                # Only evaluate bindings in a scope with a direct construction of
                # the selected class. Unrelated training code consumes no budget.
                calls = []
                for index, statement in enumerate(body):
                    expression = getattr(statement, 'value', None)
                    target = self.reader.resolve(file, expression.func) if isinstance(expression, ast.Call) else None
                    if target and (target.file, target.name) == (self.file, self.cls):
                        calls.append(index)
                if not calls:
                    continue
                env = {}
                for index, statement in enumerate(body[:calls[-1] + 1]):
                    if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return)):
                        env.clear()
                        continue
                    expression = statement.value
                    value = self.value(file, expression, env)
                    if isinstance(value, Instance) and (value.symbol.file, value.symbol.name) == (self.file, self.cls):
                        converted = self.constructor(value)
                        if converted:
                            result.append((value, converted, file, statement.lineno))
                        elif isinstance(owner, ast.Module):
                            self.unsupported_construction = True
                    elif index in calls and isinstance(owner, ast.Module):
                        self.unsupported_construction = True
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
                    for target in targets:
                        if isinstance(target, ast.Name):
                            env[target.id] = value
                        elif isinstance(target, (ast.Attribute, ast.Subscript)):
                            env.clear()
                    if isinstance(statement, ast.Expr) and isinstance(expression, ast.Call):
                        env.clear()
        return result

    def inherited_default_candidate(self):
        """Offer one no-argument call; the private worker must bind the real signature."""
        definition = self.reader.definition(self.file, self.cls)
        if (self.discovery['init'] is not None or self.discovery['examples'] or self.discovery['factories']
                or definition.decorator_list or definition.keywords or len(definition.bases) != 1
                or any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == '__init__'
                       for node in definition.body)
                or not isinstance(definition.bases[0], (ast.Name, ast.Attribute))
                or self.reader.resolve(self.file, definition.bases[0]) is not None
                or not self.reader.torch_symbol(self.file, definition.bases[0])):
            return False
        # A class reference can escape through an alias or builder argument.
        # Leave such uses explicit rather than erase their constructor intent.
        for file, tree in self.reader.trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(node.ctx, ast.Load):
                    continue
                if not self.available():
                    return False
                target = self.reader.resolve(file, node)
                if target and (target.file, target.name) == (self.file, self.cls):
                    return False
        return True

    def layer_input(self, instance, field, trail=(), hints=None):
        """Follow input shapes through known contracts into local/nested calls."""
        key = (instance.symbol.file, instance.symbol.name, field)
        if not self.available() or key in trail or len(trail) >= 12:
            return None
        file, cls = instance.symbol.file, instance.symbol.name
        definition = self.reader.definition(file, cls)
        if definition is None or not self.reader.decorators_supported(file, definition):
            return None
        env = self.parameters(instance)
        hints = {**(hints or {}), **{key: value for key, value in env.items() if _plain(value)}}
        for value in env.values():
            if isinstance(value, Instance):
                hints.update({k: v for k, v in self.attributes(value).items() if _plain(v)})
        attributes = {}
        env['self'] = attributes
        constructor = next((node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == '__init__'), None)
        if constructor is None and not any(isinstance(node, ast.FunctionDef) and node.name == 'forward' for node in definition.body) and len(definition.bases) == 1:
            base = self.reader.resolve(file, definition.bases[0])
            if base:
                return self.layer_input(Instance(Symbol(base.file, base.name), instance.args, instance.kwargs), field, (*trail, key), hints)
        for statement in constructor.body if constructor else []:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = self.value(file, statement.value, env)
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value
                    elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                        attributes[target.attr] = value
            elif isinstance(statement, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AugAssign, ast.Delete)):
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                        env[node.id] = _MISSING
                    if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)) and isinstance(node.value, ast.Name) and node.value.id == 'self':
                        attributes[node.attr] = _MISSING
            elif (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                  and len(definition.bases) == 1 and self.reader.torch_symbol(file, definition.bases[0]) == 'torch.nn.Sequential'):
                call = statement.value
                if (isinstance(call.func, ast.Attribute) and call.func.attr == '__init__'
                        and isinstance(call.func.value, ast.Call) and isinstance(call.func.value.func, ast.Name)
                        and call.func.value.func.id == 'super' and not call.keywords):
                    if statement is not constructor.body[-1]:
                        return None
                    sequence = Instance(Symbol('', 'torch.nn.Sequential'), [self.value(file, arg, env) for arg in call.args], {})
                    return self.tensor_for_layer(sequence, hints, file, call.lineno, (*trail, key))
        methods = {node.name: node for node in definition.body if isinstance(node, ast.FunctionDef)}

        def inspect_method(name, input_name, seen=()):
            method = methods.get(name)
            if (method is None or not self.reader.decorators_supported(file, method, definition)
                    or name in seen or len(seen) >= 8):
                return None
            names, found, ranks = {input_name}, [], set()
            forwarded = False

            def same_shape(node, depth=0):
                nonlocal forwarded
                if depth >= 12 or not self.available():
                    return False
                if isinstance(node, ast.Name):
                    return node.id in names
                if isinstance(node, ast.BinOp):
                    return same_shape(node.left, depth + 1) and type(_literal(node.right)) in (int, float)
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'
                        and _preserves_shape(attributes.get(node.func.attr))):
                    args = node.args + [kw.value for kw in node.keywords if kw.arg == 'input']
                    if len(args) == 1 and all(kw.arg == 'input' for kw in node.keywords) and same_shape(args[0], depth + 1):
                        forwarded = True
                        return True
                return False

            for statement in method.body:
                if isinstance(statement, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    names -= {node.id for node in ast.walk(statement) if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))}
                    continue
                expression = getattr(statement, 'value', None)
                shape = expression.func if isinstance(expression, ast.Call) and not expression.args and not expression.keywords else expression
                if (isinstance(statement, ast.Assign) and isinstance(shape, ast.Attribute)
                        and isinstance(shape.value, ast.Name) and shape.value.id in names
                        and ((shape.attr == 'shape' and not isinstance(expression, ast.Call))
                             or (shape.attr == 'size' and isinstance(expression, ast.Call)))):
                    for target in statement.targets:
                        if isinstance(target, (ast.Tuple, ast.List)) and all(isinstance(item, ast.Name) for item in target.elts):
                            ranks.add(len(target.elts))
                for call in ast.walk(expression) if expression is not None else []:
                    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name) or call.func.value.id != 'self':
                        continue
                    if name == 'forward' and any(isinstance(arg, ast.Starred) for arg in call.args):
                        from services.validation.signatures import forward_declaration
                        if forward_declaration(definition) is not method:
                            candidate = inspect_method(call.func.attr, input_name, (*seen, name))
                            if candidate:
                                found.append(candidate)
                        continue
                    layer = attributes.get(call.func.attr)
                    inputs = self.input_fields(layer) if isinstance(layer, Instance) else None
                    params = None
                    if not isinstance(layer, Instance) and call.func.attr in methods:
                        args = methods[call.func.attr].args
                        params = args.posonlyargs + args.args + args.kwonlyargs
                        inputs = [dict(name=p.arg) for p in params[1:]]
                    arguments = list(enumerate(call.args))
                    arguments.extend((i, kw.value) for kw in call.keywords for i, parameter in enumerate(inputs or [])
                                     if kw.arg == parameter['name'])
                    for position, arg in arguments:
                        if not same_shape(arg):
                            continue
                        if isinstance(layer, Instance):
                            candidate = self.tensor_for_layer(layer, hints, file, call.lineno, (*trail, key), position=position)
                            if candidate is None and layer.symbol.file:
                                if inputs and position < len(inputs):
                                    candidate = self.layer_input(layer, inputs[position]['name'], (*trail, key), hints)
                            if candidate:
                                found.append(candidate)
                        elif params is not None:
                            if position + 1 < len(params):
                                candidate = inspect_method(call.func.attr, params[position + 1].arg, (*seen, name))
                                if candidate:
                                    found.append(candidate)
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target] if isinstance(statement, ast.AnnAssign) else []
                alias = same_shape(expression)
                for target in targets:
                    names -= {node.id for node in ast.walk(target) if isinstance(node, ast.Name)}
                    if alias and isinstance(target, ast.Name):
                        names.add(target.id)
            # Linear constrains the final axis, not a particular leading rank.
            # A compatible attention/conv boundary may supply the stronger shape.
            flexible = lambda item: item[3].startswith('Linear input constraint;') and '; rank ' not in item[3]
            rigid = {repr(item[0]): item for item in found if not flexible(item)}
            if len(rigid) == 1:
                target = next(iter(rigid.values()))
                found = [target if flexible(item) and item[0]['dtype'] == target[0]['dtype']
                         and item[0]['shape'][-1] == target[0]['shape'][-1] else item for item in found]
            unique = {repr(item[0]): item for item in found}
            if len(unique) != 1 or len(ranks) > 1:
                return None
            candidate = next(iter(unique.values()))
            if ranks:
                rank = next(iter(ranks))
                if not 1 <= rank <= 6:
                    return None
                tensor, origin, line, detail = candidate
                if detail.startswith('Linear input constraint;'):
                    tensor = copy.deepcopy(tensor)
                    tensor['shape'] = ([1] + [8] * (rank - 2) if rank > 1 else []) + [tensor['shape'][-1]]
                    if math.prod(tensor['shape']) > 4_194_304:
                        return None
                    if rank == 1:
                        detail = detail.replace('; batch size 1', '')
                    detail += f'; rank {rank} from direct shape unpacking; leading extents are examples'
                    candidate = tensor, origin, line, detail
                elif rank != len(tensor['shape']):
                    return None
            if forwarded:
                candidate = (*candidate[:3], candidate[3] + '; followed known shape-preserving layers')
            return candidate

        return inspect_method('forward', field)

    def input_fields(self, layer):
        if not layer.symbol.file:
            names = {'torch.nn.MultiheadAttention': ['query', 'key', 'value'],
                     'torch.nn.Transformer': ['src', 'tgt'], 'torch.nn.TransformerEncoderLayer': ['src'],
                     'torch.nn.TransformerDecoderLayer': ['tgt', 'memory']}.get(layer.symbol.name, ['input'])
            return [dict(name=name) for name in names]
        fields = self.reader.method(layer.symbol.file, layer.symbol.name, 'forward')
        definition = self.reader.definition(layer.symbol.file, layer.symbol.name)
        if fields is None and definition and len(definition.bases) == 1 and self.reader.torch_symbol(layer.symbol.file, definition.bases[0]) == 'torch.nn.Sequential':
            return [dict(name='input')]
        return fields

    def tensor_for_layer(self, layer, hints, file, line, trail=(), *, position=0):
        if not self.available() or len(trail) >= 12:
            return None
        name = layer.symbol.name
        if not layer.symbol.file and name == 'torch.nn.Sequential' and position == 0 and layer.args and isinstance(layer.args[0], Instance):
            first = next((item for item in layer.args if not _preserves_shape(item)), None)
            if not isinstance(first, Instance):
                return None
            candidate = self.tensor_for_layer(first, hints, file, line, trail)
            if candidate is None and first.symbol.file:
                fields = self.input_fields(first)
                if fields:
                    candidate = self.layer_input(first, fields[0]['name'], trail, hints)
            return candidate
        attention = name in ('torch.nn.MultiheadAttention', 'torch.nn.Transformer',
                            'torch.nn.TransformerEncoderLayer', 'torch.nn.TransformerDecoderLayer')
        recurrent = name in ('torch.nn.RNN', 'torch.nn.GRU', 'torch.nn.LSTM')
        positions = 3 if name == 'torch.nn.MultiheadAttention' else 2 if name in ('torch.nn.Transformer', 'torch.nn.TransformerDecoderLayer') else 1
        if position >= positions:
            return None
        if layer.symbol.file or (not attention and not recurrent and name not in (
                'torch.nn.Linear', 'torch.nn.Conv1d', 'torch.nn.Conv2d', 'torch.nn.Conv3d',
                'torch.nn.ConvTranspose1d', 'torch.nn.ConvTranspose2d', 'torch.nn.ConvTranspose3d', 'torch.nn.Embedding')):
            return None
        key = ('embed_dim' if name.endswith('MultiheadAttention') else 'd_model') if attention else 'input_size' if recurrent else 'in_features' if name.endswith('Linear') else 'num_embeddings' if name.endswith('Embedding') else 'in_channels'
        width = layer.kwargs.get(key, layer.args[0] if layer.args else None)
        if name == 'torch.nn.MultiheadAttention' and position:
            override = layer.kwargs.get('kdim' if position == 1 else 'vdim', layer.args[5 + position] if len(layer.args) > 5 + position else None)
            if override is not None:
                width = override
        if type(width) is not int or not 0 < width <= 1_000_000:
            return None
        # Public constructors also accept dtype positionally. Preserve that
        # evidence rather than marking a float32 example complete for float64 weights.
        dtype_positions = {
            'torch.nn.Linear': 4, 'torch.nn.Conv1d': 10, 'torch.nn.Conv2d': 10,
            'torch.nn.Conv3d': 10, 'torch.nn.ConvTranspose1d': 11,
            'torch.nn.ConvTranspose2d': 11, 'torch.nn.ConvTranspose3d': 11,
            'torch.nn.MultiheadAttention': 10, 'torch.nn.Transformer': 14,
            'torch.nn.TransformerEncoderLayer': 10, 'torch.nn.TransformerDecoderLayer': 10,
            'torch.nn.RNN': 10, 'torch.nn.GRU': 9, 'torch.nn.LSTM': 9,
        }
        dtype_position = dtype_positions.get(name)
        dtype = layer.kwargs.get('dtype', layer.args[dtype_position]
                                 if dtype_position is not None and len(layer.args) > dtype_position else None)
        if name.endswith('Embedding'):
            # Embedding weights may be floating point; its indices are always integers.
            dtype = 'int64'
        elif dtype is None:
            dtype = 'float32'
        elif isinstance(dtype, Symbol) and not dtype.file:
            dtype = dtype.name.removeprefix('torch.')
            dtype = {'float': 'float32', 'double': 'float64', 'half': 'float16'}.get(dtype, dtype)
        else:
            return None
        assumptions = ['batch size 1', 'synthetic values']
        generator = dict(kind='normal', mean=0, std=1, seed=0)
        if attention or recurrent:
            batch_position = {'torch.nn.MultiheadAttention': 8, 'torch.nn.Transformer': 10,
                              'torch.nn.TransformerEncoderLayer': 6, 'torch.nn.TransformerDecoderLayer': 6,
                              'torch.nn.RNN': 5, 'torch.nn.GRU': 4, 'torch.nn.LSTM': 4}[name]
            batch_first = layer.kwargs.get('batch_first', layer.args[batch_position] if len(layer.args) > batch_position else False)
            if type(batch_first) is not bool:
                return None
            shape = [1, 8, width] if batch_first else [8, 1, width]
            assumptions.append('example token count 8')
        elif name.endswith('Embedding'):
            contexts = [hints[k] for k in ('block_size', 'max_seq_len', 'context_length', 'n_ctx') if type(hints.get(k)) is int and hints[k] > 0]
            shape, dtype = [1, min(contexts or [8])], 'int64'
            assumptions.append('example token count ' + str(shape[-1]))
            generator = dict(kind='randint', low=0, high=width, seed=0)
        elif name.endswith('Linear'):
            shape = [1, width]
        else:
            rank = int(name[-2])
            size = next((hints[k] for k in ('img_size', 'image_size', 'input_size') if k in hints), None)
            spatial = [size] * rank if type(size) is int else list(size) if isinstance(size, (tuple, list)) else [64] * rank
            if len(spatial) != rank or any(type(v) is not int or v <= 0 for v in spatial):
                return None
            patch = hints.get('patch_size')
            patches = [patch] * rank if type(patch) is int else patch
            if patches is not None and (not isinstance(patches, (tuple, list)) or len(patches) != rank
                    or any(type(p) is not int or p <= 0 for p in patches)):
                return None
            if patches is not None:
                if size is None:
                    spatial = [((s + p - 1) // p) * p for s, p in zip(spatial, patches)]
                elif any(s % p for s, p in zip(spatial, patches)):
                    return None
            assumptions.append('declared spatial size' if size is not None else 'example spatial size ' + ' × '.join(map(str, spatial)))
            shape = [1, width, *spatial]
        if math.prod(shape) > 4_194_304 or dtype not in ('float16', 'float32', 'float64', 'bfloat16', 'int64'):
            return None
        tensor = dict(kind='tensor', shape=shape, dtype=dtype, requires_grad=False, generator=generator)
        return tensor, file, line, f'{name.removeprefix("torch.nn.")} input constraint; ' + '; '.join(assumptions)

    def declared_inputs(self):
        def proven():
            from services.ingest.module_discovery import module_declarations
            return {(line, name) for file, line, name in module_declarations(self.reader.sources, index=self.reader.index)
                    if file == self.file}
        # The rule the drawing, the capture worker and the project graph also follow.
        declarations, stray = applicable(self.reader.sources[self.file], self.cls,
                                         tree=self.reader.trees[self.file], proven=proven)
        if not declarations and stray is not None:
            return [], ('This file has several model classes, so an input declaration outside a class applies to none of them. '
                        'Move it directly above or into its class, or use Inputs or tensorviz.toml.'), stray
        if not declarations:
            return None
        value, line = declarations[0]
        problem = 'The input declaration needs positive comma-separated dimensions (semicolon-separated tensors), at most 32 dimensions per tensor and 4,194,304 elements total. Use Inputs or tensorviz.toml for other values.'
        if len(declarations) != 1:
            return [], 'Several input declarations were found. Choose one input setup in Inputs or tensorviz.toml.', line
        if len(value) > 4096 or len(value.split(';')) > 64:
            return [], problem, line
        shapes = []
        for item in value.split(';'):
            if not re.fullmatch(r'[0-9]+(?:\s*,\s*[0-9]+)*', item.strip()):
                return [], problem, line
            shape = [int(dimension) for dimension in item.split(',')]
            if len(shape) > 32 or any(not 0 < dimension <= 2**31 - 1 for dimension in shape):
                return [], problem, line
            shapes.append(shape)
        if sum(math.prod(shape) for shape in shapes) > 4_194_304:
            return [], problem, line
        return [dict(kind='tensor', shape=shape, dtype='float32', requires_grad=False,
                     generator=dict(kind='normal', mean=0, std=1, seed=0)) for shape in shapes], None, line

    def resolve(self):
        discovery = self.discovery
        declared = self.declared_inputs()
        profile = dict(kind='model', model=discovery['model'], mode='eval', device='cpu', seed=0,
                       grad_enabled=False, init=dict(args=[], kwargs={}), inputs=dict(args=[], kwargs={}))
        draft, unresolved = dict(profile=profile), []
        complete_examples = {repr((item['init'], item['inputs'])): item for item in discovery['examples'] if 'inputs' in item}
        if len(complete_examples) == 1:
            example = next(iter(complete_examples.values()))
            profile.update(init=copy.deepcopy(example['init']), inputs=copy.deepcopy(example['inputs']))
            self.note('call', 'repository', 'Constructor and input recipe from this example call; tensor values are recreated.', example['file'], example['line'])
        elif len(complete_examples) > 1:
            unresolved.append('Several example calls use different settings. Choose the example you want to capture.')
        construction = Instance(Symbol(self.file, self.cls), [], {})
        if not complete_examples:
            choices = {repr(item[1]): item for item in self.source_constructions()}
            concrete = {key: item for key, item in choices.items() if not item[1][2] and not item[1][3]}
            if concrete:
                choices = concrete
            if len(choices) == 1:
                construction, (binding, config, unknown, defaults), file, line = next(iter(choices.values()))
                profile['init'] = binding
                unresolved.extend(f'init.{name}: the source call uses a value that needs configuration.' for name in unknown)
                if config:
                    draft['configObject'] = config
                self.note('init', 'repository', 'Constructor arguments from this source call.', file, line)
                for name in defaults:
                    self.note('init.' + name, 'example',
                              'The source-call value is unresolved; this example uses the declared Python default.', file, line)
            elif len(choices) > 1:
                unresolved.append('Several constructor configurations were found. Choose one before capturing.')
            elif self.unsupported_construction:
                unresolved.append('A source constructor call needs explicit configuration. Review its arguments before capturing.')
            if 'configObject' not in draft:
                configs = [(i, item) for i, item in enumerate(discovery['configs']) if item['parameter'] and self.config_supported(item)
                           and item['fields'] is not None]
                missing = [f for f in discovery['init'] or [] if f['required']]
                if (len(configs) == 1 and configs[0][1]['parameter'] in {field['name'] for field in missing}
                        and self.parameters(construction).get(configs[0][1]['parameter'], _MISSING) is _MISSING):
                    index, config = configs[0]
                    draft['configObject'] = dict(index=index, parameter=config['parameter'], values={})
                    construction.kwargs[config['parameter']] = Instance(Symbol(config['file'], config['cls']), [], {})
                    self.note('init.' + config['parameter'], 'source', 'Configuration type from the constructor annotation; declared defaults are preserved.', config['file'],
                              self.reader.definition(config['file'], config['cls']).lineno)
            if len(choices) <= 1 and not self.unsupported_construction and not discovery['factories']:
                from .graph_setup_constraints import propose
                candidate, notes = propose(self, construction)
                converted = self.constructor(candidate) if notes else None
                if converted:
                    binding, config, unknown, _ = converted
                    construction = candidate
                    profile['init'] = binding
                    if config:
                        draft['configObject'] = config
                    # Filling an earlier positional-only gap can also make a
                    # later, already-known source argument serializable.
                    unresolved = [reason for reason in unresolved
                                  if not reason.startswith('init.') or reason.split(':', 1)[0][5:] in unknown]
                    for name, value, file, line, detail in notes:
                        self.note(name, 'example', f'Example value {value} from {detail} dimension constraints; editable, not a recovered configuration.', file, line)
            for field in discovery['inputs'] or []:
                if field['required'] and field['kind'] not in ('varargs', 'kwargs'):
                    found = self.layer_input(construction, field['name'])
                    if found:
                        tensor, file, line, detail = found
                        if field['kind'] == 'positional':
                            preceding = [f for f in discovery['inputs'][:discovery['inputs'].index(field)] if f['kind'] == 'positional']
                            if len(profile['inputs']['args']) != len(preceding):
                                continue
                            profile['inputs']['args'].append(tensor)
                        else:
                            profile['inputs']['kwargs'][field['name']] = tensor
                        self.note('inputs.' + field['name'], 'example', detail, file, line)
        if declared is not None:
            tensors, problem, line = declared
            profile['inputs'] = dict(args=tensors, kwargs={})
            self.evidence = [entry for entry in self.evidence if not entry['field'].startswith('inputs.')]
            for entry in self.evidence:
                if entry['field'] == 'call':
                    entry.update(field='init', detail='Constructor recipe from this source call; inputs use the declaration.')
            if problem:
                unresolved.append('inputs: ' + problem)
            else:
                self.note('inputs', 'source', 'Explicit input declaration; positional float32 tensors with normal values and seed 0.', self.file, line)
                fields = discovery['inputs']
                if fields is not None and not any(field['kind'] == 'varargs' for field in fields):
                    if len(tensors) > sum(field['kind'] in ('positional', 'keyword') for field in fields):
                        unresolved.append('inputs: The input declaration supplies more positional tensors than forward accepts.')
        for group in ('init', 'inputs'):
            fields = discovery[group]
            if fields is None:
                unresolved.append(f'{group}: the Python signature could not be resolved.')
                continue
            positional = [f for f in fields if f['kind'] in ('positional', 'keyword')]
            binding = profile[group]
            for field in fields:
                configured = field['name'] in binding['kwargs'] or field in positional[:len(binding['args'])]
                if group == 'init' and draft.get('configObject', {}).get('parameter') == field['name']:
                    configured = True
                    option = discovery['configs'][draft['configObject']['index']]
                    for config_field in option['fields'] or []:
                        if config_field['required'] and config_field['name'] not in draft['configObject']['values']:
                            unresolved.append(f"config.{config_field['name']}: provide a value.")
                if field['required'] and not configured:
                    unresolved.append(f"{group}.{field['name']}: provide a value.")
                if not field['required'] and field['kind'] not in ('varargs', 'kwargs') and not configured:
                    self.note(group + '.' + field['name'], 'source', 'Uses the Python default: ' + field.get('defaultText', 'declared default'), self.file,
                              self.reader.definition(self.file, self.cls).lineno)
            if any(f['kind'] in ('varargs', 'kwargs') for f in fields) and not complete_examples:
                unresolved.append(f'{group}: dynamic arguments need an explicit example call.')
        factories = [factory for factory in discovery['factories']
                     if factory['file'].rsplit('/', 1)[-1] != 'setup.py'
                     and all(not field['required'] for field in factory['fields'])]
        factories.sort(key=lambda factory: (factory['callable'].startswith('_'), factory['file'], factory['line']))
        factory = factories[0] if factories else None
        construction_missing = any(reason.startswith(('init:', 'init.', 'config.')) for reason in unresolved)
        known_init = bool(profile['init']['args'] or profile['init']['kwargs']
                          or draft.get('configObject', {}).get('values'))
        if (not complete_examples and construction_missing and factory and not known_init
                and all(reason.startswith(('init:', 'init.', 'config.', 'inputs:', 'inputs.')) for reason in unresolved)):
            # A factory owns construction. Do not carry input dimensions inferred
            # from an incomplete direct constructor into its potentially different model.
            profile.pop('init')
            draft.pop('configObject', None)
            profile['factory'] = dict(file=factory['file'], callable=factory['callable'], kwargs={})
            if declared is None:
                profile['inputs'] = dict(args=[], kwargs={})
                unresolved = [reason for reason in unresolved if reason.startswith('inputs:')]
                unresolved.extend(f"inputs.{field['name']}: provide a value." for field in discovery['inputs'] or []
                                  if field['required'])
            else:
                unresolved = [reason for reason in unresolved if reason.startswith(('inputs:', 'inputs.'))]
            self.evidence = [entry for entry in self.evidence if not entry['field'].startswith(('init', 'inputs.'))]
            self.note('factory', 'example', 'A repository factory callable without arguments supplies this example variant; its result must pass capture.',
                      factory['file'], factory['line'])
        inherited_default = (not complete_examples and not draft.get('configObject')
                             and profile.get('init') == dict(args=[], kwargs={})
                             and profile['inputs'] == dict(args=[], kwargs={})
                             and self.inherited_default_candidate())
        if inherited_default:
            unresolved = [reason for reason in unresolved if not reason.startswith('init:')]
            self.note('init', 'example',
                      'Try the inherited constructor without arguments; the worker must verify that its defaults are sufficient.',
                      self.file, self.reader.definition(self.file, self.cls).lineno)

        def tensor_elements(value):
            if isinstance(value, dict):
                if value.get('kind') == 'tensor':
                    return math.prod(value['shape'])
                return sum(tensor_elements(item) for item in value.values())
            return sum(tensor_elements(item) for item in value) if isinstance(value, list) else 0

        if tensor_elements(profile) > 4_194_304:
            unresolved.append('The example exceeds the automatic input size limit. Review its dimensions before capturing.')
        if discovery['truncated'] or self.steps > 20_000 or len(discovery['examples']) >= 64:
            unresolved.append('Source discovery was incomplete. Review the proposed example before capturing.')
        self.note('execution', 'example', 'CPU evaluation, gradients off and seed 0 are example execution settings.')
        # Optional library inputs can still be semantically required (e.g. tokens
        # or pixels). Ask the constructed model for its contract before saving an
        # empty call. Known zero-input declarations and complete calls need no probe.
        runtime_inputs = (declared is None and not complete_examples and (inherited_default or discovery['inputs'] != [])
                          and not draft.get('configObject') and profile['inputs'] == dict(args=[], kwargs={})
                          and all(reason.startswith(('inputs:', 'inputs.')) for reason in unresolved))
        if runtime_inputs and not unresolved:
            unresolved.append('inputs: check the constructed model for an example input contract.')
        result = dict(status='partial' if unresolved else 'complete', draft=draft, evidence=self.evidence, unresolved=unresolved[:64])
        if runtime_inputs:
            result['runtimeInputs'] = True
        return result


def resolve_setup(reader, file, cls, discovery):
    return Resolver(reader, file, cls, discovery).resolve()

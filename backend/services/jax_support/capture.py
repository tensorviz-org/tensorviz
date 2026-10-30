"""Translate Jaxpr values and equations into the shared, bounded graph contract.

These are abstract program facts, never dispatcher events or runtime timings.
Only unconditional call bodies are expanded; control-flow primitives retain
their boundary and explicitly disclose unexpanded bodies.
"""
from pathlib import Path
import re
from uuid import uuid4

from ..semantic.contracts import CaptureSnapshot
from ..semantic.execution_plan import applicability_digest
from ..validation.setup import ExecutionRequirement
from .setup import callable_in_trace, resolve_device


class JaxprCapture:
    def __init__(self, plan, workspace, state):
        import jax
        self.plan, self.workspace, self.state = plan, Path(workspace).resolve(), state
        self.sequence = 0
        self.omissions = set()
        self.source_lines = {}
        self.bindings = set()
        self.source_files = {file.path: file.digest for file in plan.source.files if file.role == 'source'}
        self.data = dict(version=1, id=f'capture:{uuid4()}', attemptId=plan.id,
            source=plan.source.model_dump(mode='json'), planDigest=applicability_digest(plan),
            preparationDigest=None, automatic=None, inputDigest=None, stateDigest=None,
            backend='jaxpr', backendVersion='1', torchVersion=None, jaxVersion=jax.__version__,
            normalizations=['Jaxpr abstract tracing; tensor shapes are inferred, not measured runtime values',
                'Python scalar inputs are static; array leaves are traced',
                'Custom PyTrees normalize to registered child keys; static PyTree metadata is not serialized',
                'Unconditional JIT, checkpoint and custom derivative call bodies are expanded; control-flow bodies remain opaque',
                'Primitive named tuples normalize to tuples; primitive enums serialize by qualified member name',
                'No storage aliases, peak memory, or eager execution coverage is claimed'],
            status='complete', rootInvocationId='call:0', inputs=dict(args=[], kwargs={}), output=None,
            definitions=[], modules=[], invocations=[], operations=[], values=[], sourceBindings=[], diagnostics=[],
            coverage=dict(scope='traced_program', operationsSeen=0, invocationsSeen=0, omissions=[], modules=[]),
            limits=plan.captureLimits.model_dump())

    def tick(self):
        self.sequence += 1
        return self.sequence

    def literal(self, value):
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
        if value is None:
            return dict(kind='none')
        if type(value) is bool:
            return dict(kind='bool', value=value)
        if type(value) is int and len(str(value)) <= 4096:
            return dict(kind='int', value=str(value))
        if type(value) is float:
            import math
            encoded = ('nan' if math.isnan(value) else 'inf' if value == math.inf else '-inf'
                       if value == -math.inf else '-0' if value == 0 and math.copysign(1, value) < 0 else value)
            return dict(kind='float', value=encoded)
        if type(value) is str and len(value) <= 4096:
            return dict(kind='str', value=value)
        if isinstance(value, np.dtype):
            return dict(kind='dtype', value=str(value))
        from enum import Enum
        import jax
        if isinstance(value, getattr(jax.lax, 'RaggedDotDimensionNumbers', ())):
            from dataclasses import fields
            return self.literal({field.name: getattr(value, field.name) for field in fields(value)})
        if isinstance(value, Enum):
            return self.literal(f'{type(value).__name__}.{value.name}')
        if isinstance(value, (tuple, list)) and len(value) <= 4096:
            return dict(kind='tuple' if isinstance(value, tuple) else 'list', items=[self.literal(item) for item in value])
        if type(value) is dict and len(value) <= 256 and all(type(key) is str for key in value):
            return dict(kind='dict', entries=[dict(key=self.literal(key), value=self.literal(item)) for key, item in value.items()])
        self.omissions.add('unsupported_type')
        return dict(kind='opaque', reason='unsupported_type', typeName=None)

    def value(self, aval, origin, *, producer=None, actual=None):
        if len(self.data['values']) >= self.plan.captureLimits.values:
            raise ExecutionRequirement('forward', 'capture_limited', ['limits', 'max_tensors'], 'adjust_limits')
        name = f'value:{len(self.data["values"])}'
        shape = getattr(aval, 'shape', None)
        shape = [int(dim) for dim in shape] if shape is not None else None
        state = [self.state[id(actual)]] if actual is not None and id(actual) in self.state else []
        self.data['values'].append(dict(id=name, objectId=name, storageId=None, version=None,
            shape=shape, dtype=str(getattr(aval, 'dtype', 'opaque')), device='abstract', requiresGrad=False,
            aliasOf=None, previousVersion=None, logicalVersion=0, stride=None, storageOffset=None,
            logicalBytes=None, producer=producer, origin='state' if state else origin, state=state))
        return dict(kind='value', id=name)

    def enter(self, name, arguments, parent=None):
        if len(self.data['invocations']) >= self.plan.captureLimits.invocations:
            raise ExecutionRequirement('forward', 'capture_limited', ['limits', 'max_events'], 'adjust_limits')
        index = len(self.data['invocations'])
        definition, module, call = f'definition:{index}', f'module:{index}', f'call:{index}'
        self.data['definitions'].append(dict(id=definition, qualifiedName=name[:512], sourceBindingIds=[]))
        self.data['modules'].append(dict(id=module, definitionId=definition, paths=[]))
        record = dict(id=call, moduleId=module, parentId=parent, sequence=self.tick(), endSequence=None,
                      arguments=arguments, output=None, status='completed', diagnosticId=None,
                      training=False, gradEnabled=False, sourceBindingIds=[])
        self.data['invocations'].append(record)
        return record

    def source(self, equation):
        import hashlib
        info = getattr(equation, 'source_info', None)
        trace = getattr(info, 'traceback', None)
        files = self.source_files
        for frame in reversed(getattr(trace, 'frames', ())):
            path = Path(frame.file_name).resolve()
            if not path.is_relative_to(self.workspace):
                continue
            file = path.relative_to(self.workspace).as_posix()
            if file not in files:
                continue
            if file not in self.source_lines:
                raw = path.read_bytes()
                # Frame line numbers count only \r\n, \r and \n, as the AST does.
                self.source_lines[file] = re.split(r'\r\n?|\n', raw.decode('utf-8')) if hashlib.sha256(raw).hexdigest() == files[file] else []
            lines = self.source_lines[file]
            line = frame.line_num
            if not 1 <= line <= len(lines) or not lines[line - 1]:
                continue
            identity = f'jax-source:{file}:{line}'
            if identity not in self.bindings:
                self.bindings.add(identity)
                self.data['sourceBindings'].append(dict(id=identity, snapshotId=self.plan.source.id,
                    kind='call', quality='candidate', anchors=[dict(file=file, digest=files[file],
                        startLine=line, startColumn=0, endLine=line, endColumn=len(lines[line - 1].encode('utf-16-le')) // 2,
                        scope=frame.function_name[:4096], tokenDigest=hashlib.sha256(lines[line - 1].encode()).hexdigest())]))
            return [identity]
        return []

    def program(self, closed, inputs, owner, depth=0):
        if depth > 32:
            raise ValueError('JAX call nesting exceeds 32 levels.')
        # JAX <= 0.8 uses ClosedJaxpr; newer releases merge it into Jaxpr.
        program = getattr(closed, 'jaxpr', closed)
        constants = getattr(closed, 'consts', ())
        if len(program.invars) != len(inputs) or len(program.constvars) != len(constants):
            raise ValueError('Unsupported Jaxpr argument/constant binding.')
        values = dict(zip(program.invars, inputs))
        for var, actual in zip(program.constvars, constants):
            values[var] = self.value(var.aval, 'external', actual=actual)

        def read(var):
            if hasattr(var, 'val'):
                import numpy as np
                literal = np.asarray(var.val)
                return self.literal(literal.item()) if literal.ndim == 0 else self.value(var.aval, 'external', actual=var.val)
            return values[var]

        for equation in program.eqns:
            args = [read(var) for var in equation.invars]
            name = equation.primitive.name
            body = next((equation.params[key] for key in ('jaxpr', 'call_jaxpr', 'fun_jaxpr')
                         if key in equation.params), None)
            can_expand = name in {'jit', 'pjit', 'xla_call', 'custom_jvp_call', 'custom_vjp_call', 'closed_call', 'call', 'named_call', 'remat2'}
            if body is not None and can_expand and not equation.effects:
                nested = getattr(body, 'jaxpr', body)
                if len(getattr(nested, 'invars', ())) == len(args):
                    label = equation.params.get('name', name)
                    # Compiled NNX programs flatten hundreds of parameter leaves
                    # into operands. Preserve them as one ordered operand tuple
                    # when they exceed the Python-call positional-argument bound.
                    arguments = [dict(kind='tuple', items=args)] if len(args) > 256 else args
                    call = self.enter(f'jax.{label}', dict(args=arguments, kwargs={}), owner)
                    call['sourceBindingIds'] = self.source(equation)
                    outputs = self.program(body, args, call['id'], depth + 1)
                    call.update(output=dict(kind='tuple', items=outputs), endSequence=self.tick())
                    values.update(zip(equation.outvars, outputs))
                    continue
            if len(self.data['operations']) >= self.plan.captureLimits.operations:
                raise ExecutionRequirement('forward', 'capture_limited', ['limits', 'max_events'], 'adjust_limits')
            identity = f'op:{len(self.data["operations"])}'
            keywords = {}
            for key, item in equation.params.items():
                if hasattr(item, 'eqns') or hasattr(item, 'jaxpr') or (isinstance(item, (list, tuple))
                        and any(hasattr(child, 'eqns') or hasattr(child, 'jaxpr') for child in item)):
                    self.omissions.add('opaque_kernel')
                    keywords[key] = dict(kind='opaque', reason='opaque_kernel', typeName=None)
                else:
                    keywords[key] = self.literal(item)
            if equation.effects:
                self.omissions.add('opaque_kernel')
                keywords['jax_effects'] = dict(kind='opaque', reason='opaque_kernel', typeName=None)
            outputs = []
            for index, var in enumerate(equation.outvars):
                ref = self.value(var.aval, 'operation', producer=dict(operationId=identity, kind='output', path=['items', index]))
                outputs.append(ref)
                values[var] = ref
            self.data['operations'].append(dict(id=identity, sequence=self.tick(), invocationId=owner,
                target='jax.' + name, schemaSignature=None, arguments=dict(args=args, kwargs=keywords),
                output=dict(kind='tuple', items=outputs), effects=[], status='completed', diagnosticId=None,
                sourceBindingIds=self.source(equation)))
        return [read(var) for var in program.outvars]

    def capture(self, function, args, kwargs):
        import jax
        import numpy as np
        leaves, structure = jax.tree_util.tree_flatten((args, kwargs))
        positions = [index for index, item in enumerate(leaves) if isinstance(item, (jax.Array, np.ndarray))]
        arrays = [leaves[index] for index in positions]

        def traced(*inputs):
            actual = list(leaves)
            for index, array in zip(positions, inputs):
                actual[index] = array
            positional, keywords = jax.tree_util.tree_unflatten(structure, actual)
            return callable_in_trace(function)(*positional, **keywords)

        with jax.default_device(resolve_device(self.plan.execution['device'])):
            program, output_shapes = jax.make_jaxpr(traced, return_shape=True)(*arrays)
        raw = getattr(program, 'jaxpr', program)
        inputs = [self.value(var.aval, 'input') for var in raw.invars]
        encoded = [self.literal(item) if index not in positions else inputs[positions.index(index)]
                   for index, item in enumerate(leaves)]
        def tree(item, references, depth=0):
            if depth > 32:
                raise ValueError('JAX argument nesting exceeds 32 levels.')
            if item is None:
                return dict(kind='none')
            # Flatten one level without reconstructing user containers with
            # encoded references (constructors may enforce array invariants).
            children, definition = jax.tree_util.tree_flatten_with_path(
                item, is_leaf=lambda child: child is not item)
            if definition.node_data() is None:
                return next(references)
            if len(children) > 4096:
                raise ValueError('JAX argument container exceeds 4096 children.')
            if type(item) in (tuple, list):
                return dict(kind=type(item).__name__, items=[tree(child, references, depth + 1)
                    for _, child in children])
            entries = []
            for path, child in children:
                key = path[0]
                key = getattr(key, 'name', getattr(key, 'key', getattr(key, 'idx', None)))
                encoded_key = self.literal(key)
                if encoded_key['kind'] not in ('none', 'bool', 'int', 'str'):
                    raise ValueError('JAX PyTree keys must be literal strings or integers.')
                entries.append(dict(key=encoded_key, value=tree(child, references, depth + 1)))
            return dict(kind='dict', entries=entries)

        references = iter(encoded)
        boundaries = dict(args=[tree(item, references) for item in args],
            kwargs={key: tree(kwargs[key], references) for key in sorted(kwargs)})
        root = self.enter(self.plan.source.className, boundaries)
        outputs = self.program(program, inputs, root['id'])
        output = tree(output_shapes, iter(outputs))
        root.update(output=output, endSequence=self.tick())
        self.data.update(inputs=boundaries, output=output, status='partial' if self.omissions else 'complete')
        self.data['coverage'].update(operationsSeen=len(self.data['operations']),
            invocationsSeen=len(self.data['invocations']), omissions=sorted(self.omissions))
        from pydantic import ValidationError
        try:
            return CaptureSnapshot.model_validate(self.data).model_dump(mode='json')
        except ValidationError as error:
            if any(str(item.get('ctx', {}).get('error')) == 'capture exceeds byte budget'
                   for item in error.errors(include_input=False)):
                raise ExecutionRequirement('forward', 'capture_limited', ['limits', 'max_capture_bytes'], 'adjust_limits') from error
            raise

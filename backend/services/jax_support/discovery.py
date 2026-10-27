"""Conservative, nonexecuting discovery of JAX functions and Flax classes."""
import ast

from services.ide.source_decorators import imported_symbol
from services.ingest.module_discovery import module_declarations

FLAX_BASES = {'flax.linen.Module', 'flax.linen.module.Module',
              'flax.nnx.Module', 'flax.nnx.module.Module'}


def declarations(sources):
    found = module_declarations(sources, framework_bases=FLAX_BASES)
    for file, source in sources.items():
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError):
            continue
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        for function in functions:
            if sum(node.name == function.name for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))) != 1:
                continue
            local_names = {node.arg for node in ast.walk(function) if isinstance(node, ast.arg)}
            local_names.update(node.id for node in ast.walk(function) if isinstance(node, ast.Name)
                               and isinstance(node.ctx, (ast.Store, ast.Del)))
            # Include functions that use an imported JAX operation or transform.
            # Merely importing JAX must not turn every Python helper into a model.
            calls = [node.func for node in ast.walk(function) if isinstance(node, ast.Call)]
            calls += [node.func if isinstance(node, ast.Call) else node for node in function.decorator_list]
            if any(ast.unparse(call).split('.')[0] not in local_names
                   and (imported_symbol(tree, call) or '').startswith('jax.') for call in calls):
                found.add((file, function.lineno, function.name))
    return found


def target(sources, file, name):
    try:
        tree = ast.parse(sources[file])
    except (SyntaxError, ValueError, RecursionError):
        return None
    matches = [node for node in tree.body
               if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
    return (matches[0] if len(matches) == 1 and
            (file, matches[0].lineno, name) in declarations(sources) else None)


def discover(sources, file, name, digest):
    from services.ide.graph_setup import _parameters, _literal, _MISSING
    from services.ide.module_index import module_of_file
    definition = target(sources, file, name)
    if definition is None:
        return None
    if isinstance(definition, ast.FunctionDef):
        init, inputs = [], _parameters(definition, method=False, decorators_known=True)
    else:
        constructor = next((node for node in definition.body if isinstance(node, ast.FunctionDef)
                            and node.name == '__init__'), None)
        call = next((node for node in definition.body if isinstance(node, ast.FunctionDef)
                     and node.name == '__call__'), None)
        inputs = _parameters(call, decorators_known=True)
        init = _parameters(constructor, decorators_known=True) if constructor else []
        if constructor is None:
            # Linen's constructor fields are dataclass annotations. Inherited
            # fields remain runtime-checked; never invent a value for them.
            for field in definition.body:
                if not isinstance(field, ast.AnnAssign) or not isinstance(field.target, ast.Name):
                    continue
                entry = dict(name=field.target.id, kind='keyword', required=field.value is None)
                if field.value is not None:
                    value = _literal(field.value)
                    entry['defaultText'] = ast.unparse(field.value)[:160]
                    if value is not _MISSING:
                        entry['defaultValue'] = value
                init.append(entry)
    profile = dict(kind='model', framework='jax', model=f'{module_of_file(file)}:{name}',
                   mode='eval', device='cpu', grad_enabled=False, seed=0,
                   init=dict(args=[], kwargs={}), inputs=dict(args=[], kwargs={}))
    unresolved = []
    for field, entries in [('init', init), ('inputs', inputs)]:
        if entries is None:
            unresolved.append(f'{field}: provide an explicit binding or Python factory.')
            continue
        for entry in entries:
            if entry['required']:
                unresolved.append(f"{field}.{entry['name']}: provide a value.")
    return dict(model=profile['model'], init=init, inputs=inputs, examples=[], factories=[], configs=[],
                parents=[], constraints=[], truncated=False, sourceDigest=digest,
                resolution=dict(status='partial' if unresolved else 'complete',
                    draft=dict(profileName=('tensorviz_' + name)[:64], profile=profile),
                    unresolved=unresolved, evidence=[]))

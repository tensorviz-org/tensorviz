"""Recognize source tensor layout calls without evaluating their shape expressions."""
import ast


TORCH_LAYOUT_TARGETS = {
    'torch.reshape', 'torch.permute', 'torch.nn.functional.interpolate',
}


def tensor_layout_call(call):
    """Return (build info, tensor operand); scalar arguments are not tensor edges.

    Literal options can be consumed by the existing operation registry. Symbolic
    options remain source-only metadata, never fabricated sizes or runtime inputs.
    """
    target = getattr(call, '_torch_target', None)
    method = (isinstance(call.func, ast.Attribute) and not target
              and call.func.attr in ('reshape', 'view', 'permute'))
    if target not in TORCH_LAYOUT_TARGETS and not method:
        return None
    name = call.func.attr if method else target.rsplit('.', 1)[-1]
    if method and name in ('reshape', 'view') and len(call.args) == 2:
        # Preserve the existing flatten-before-Linear lowering.
        if ast.unparse(call.args[1]) == '-1':
            return None
    keywords = {item.arg: item.value for item in call.keywords if item.arg}
    tensor = call.func.value if method else call.args[0] if call.args else keywords.get('input')
    if tensor is None:
        return None
    args = call.args if method else call.args[1:]
    options = {}
    if name in ('reshape', 'view', 'permute'):
        key = 'dims' if name == 'permute' else 'shape'
        if args:
            options[key] = args[0] if len(args) == 1 else ast.Tuple(elts=args, ctx=ast.Load())
        elif key in keywords or name == 'view' and 'size' in keywords:
            options[key] = keywords.get(key, keywords.get('size'))
        else:
            return None
        # Tensor.view(dtype) is a different overload, not a shape change.
        if name == 'view' and (isinstance(options[key], ast.Attribute)
                               and ast.unparse(options[key]).startswith('torch.')):
            return None
        target = 'torch.permute' if name == 'permute' else 'torch.reshape'
    else:
        target = 'interpolate'  # Registry/codegen canonical; emitted from torch.nn.functional.
        names = ('size', 'scale_factor', 'mode', 'align_corners', 'recompute_scale_factor', 'antialias')
        options.update(zip(names, args))
        options.update((key, value) for key, value in keywords.items() if key != 'input')
    parameters, expressions = {}, {}
    for key, value in options.items():
        try:
            literal = ast.literal_eval(value)
            if isinstance(literal, tuple):
                literal = list(literal)
            if not (literal is None or type(literal) in (str, bool, int, float)
                    or isinstance(literal, list) and all(type(item) in (str, bool, int, float) for item in literal)):
                raise ValueError('not a scalar option')
            parameters[key] = literal
        except (ValueError, TypeError):
            expressions[key] = ast.unparse(value)
    unpacked = [ast.unparse(item.value) for item in call.keywords if item.arg is None]
    if unpacked:
        expressions['**kwargs'] = ', '.join(unpacked)
    if expressions:
        parameters['_source_arguments'] = expressions
    return {'emit': 'functional', 'type': target, 'parameters': parameters}, tensor

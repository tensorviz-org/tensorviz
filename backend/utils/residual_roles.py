"""Conservative skip-path annotations for the static parser, without execution.

Only straight-line, single-input forwards qualify. An identity bypass must feed
an add alongside a descendant of itself. A projection shortcut additionally
needs a source-declared `shortcut`/`downsample` 1x1 convolution or linear layer
and a shared branch input. Arbitrary parallel sums stay unclassified.
"""
import ast
from collections import defaultdict


def annotate_residual_edges(nodes, edges, forward, aliases, known_call):
    if forward is None:
        return
    args = forward.args
    if len(args.posonlyargs + args.args + args.kwonlyargs) != 2 or args.vararg or args.kwarg:
        return
    if any(isinstance(n, (ast.If, ast.IfExp, ast.For, ast.While, ast.Try, ast.With))
           for n in ast.walk(forward)):
        return
    # The parser deliberately passes through some unmodelled expressions. They
    # cannot prove an *identity* bypass (e.g. x[:, :4] or mystery(x)). Fail closed
    # instead of inferring meaning from that lossy topology.
    calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call)]
    callees = {id(call.func) for call in calls}
    for expr in ast.walk(forward):
        if isinstance(expr, (ast.Subscript, ast.UnaryOp, ast.BoolOp)):
            return
        if isinstance(expr, ast.BinOp) and not isinstance(expr.op, ast.Add):
            return
        if isinstance(expr, ast.Attribute) and id(expr) not in callees:
            return
    module_calls = []
    for call in calls:
        func = call.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'self' and func.attr in aliases:
            module_calls.append(func.attr)
        elif not known_call(call):
            return
    if len(module_calls) != len(set(module_calls)):
        return  # one canvas node cannot disambiguate repeated calls
    by_id = {node.id: node for node in nodes}
    incoming = defaultdict(list)
    for edge in edges:
        incoming[edge.target].append(edge)

    def ancestors(node_id, blocked):
        seen, queue = {}, [(node_id, 0)]
        while queue:
            current, distance = queue.pop()
            if current in seen or current == blocked:
                continue
            seen[current] = distance
            node = by_id.get(current)
            # Opaque code and merges cannot establish a single bypass origin.
            if node is None or node.data.layerType.id in ('python', 'custom_code'):
                continue
            parents = incoming[current]
            if len(parents) == 1:
                queue.append((parents[0].source, distance + 1))
        return seen

    for node in nodes:
        inputs = incoming[node.id]
        if node.data.layerType.id != 'add' or len(inputs) != 2:
            continue
        candidates = []
        for bypass, transformed in (inputs, inputs[::-1]):
            origin = by_id.get(bypass.source)
            if origin is None:
                continue
            if bypass.source in ancestors(transformed.source, node.id):
                candidates.append((bypass, 'identity-bypass'))
                continue
            kind = origin.data.layerType.id.lower()
            name = origin.id.rsplit('.', 1)[-1].lower()
            kernel = origin.data.parameters.get('kernel_size')
            projection = kind == 'linear' or (
                kind in ('conv1d', 'conv2d', 'conv3d')
                and (kernel == 1 or isinstance(kernel, (list, tuple)) and kernel and all(k == 1 for k in kernel))
            )
            parents = incoming[bypass.source]
            if name in ('shortcut', 'downsample') and projection and len(parents) == 1:
                if ancestors(transformed.source, node.id).get(parents[0].source, 0) >= 2:
                    candidates.append((bypass, 'projection-shortcut'))
        if len(candidates) == 1:
            edge, evidence = candidates[0]
            edge.data = {'role': 'residual', 'roleEvidence': evidence}

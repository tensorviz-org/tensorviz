"""Reversible views of captured records, with no source execution or layout edges.

Every retained operation and invocation survives collapse. Dataflow comes only
from Value.producer and a particular argument/result/effect use. Invocation ports
describe observed interfaces; ownership does not create a second computation.
"""
from collections import defaultdict
import hashlib
import json

from .contracts import (
    CaptureProjection, CaptureSnapshot, MappingArgument, ProjectionEdge, ProjectionNode,
    ProjectionPort, SequenceArgument, ValueRef, walk_argument, walk_arguments,
)


def _id(kind, *parts):
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return kind + ":" + hashlib.sha256(encoded).hexdigest()


def _leaves(tree):
    for path, argument in tree:
        if isinstance(argument, SequenceArgument) and argument.items:
            continue
        if isinstance(argument, MappingArgument) and argument.entries:
            continue
        yield path, argument


def project_capture(payload, *, collapsed_invocations=()):
    """Project one validated capture; display grouping never modifies its records.

    Repeated operands have distinct destination ports. Repeated module calls have
    distinct regions even when they share one module instance. Empty/identity
    modules retain their measured interfaces without fabricated operator edges.
    """
    capture = CaptureSnapshot.model_validate(payload)
    collapsed = set(collapsed_invocations)
    if not collapsed <= {call.id for call in capture.invocations}:
        raise ValueError("cannot collapse an invocation outside this capture")
    # Place retained state/external sources at the common scope of their uses.
    # This is display containment, not a claim that a call owns a parameter.
    # Shared values remain outside separate calls instead of being duplicated.
    parents = {call.id: call.parentId for call in capture.invocations}
    depths = {}
    for call in capture.invocations:
        depths[call.id] = 1 + depths.get(call.parentId, 0)

    def common_scope(left, right):
        while left != right:
            if left is None or right is None:
                return None
            if depths[left] >= depths[right]:
                left = parents[left]
            else:
                right = parents[right]
        return left

    value_scopes = {}

    def use(value, scope):
        value_scopes[value] = common_scope(value_scopes[value], scope) if value in value_scopes else scope

    def tree_uses(tree, scope):
        for _, argument in tree:
            if isinstance(argument, ValueRef):
                use(argument.id, scope)

    for op in capture.operations:
        tree_uses(walk_arguments(op.arguments), op.invocationId)
        for effect in op.effects:
            use(effect.beforeId, op.invocationId)
    for call in capture.invocations:
        tree_uses(walk_arguments(call.arguments), call.id)
        if call.output is not None:
            tree_uses(walk_argument(call.output), call.id)
    if capture.output is not None:
        tree_uses(walk_argument(capture.output), None)
    nodes = []
    ports = []
    edges = []
    owners = {}
    node_map = {}
    port_map = {}
    port_paths = {}

    def node(kind, semantic_id, parent=None):
        identity = _id(kind, capture.id, semantic_id)
        parent_node = node_map.get(parent)
        owner = (owners[parent] or (parent if parent_node.collapsed else None)) if parent is not None else None
        item = ProjectionNode(id=identity, kind=kind, semanticId=semantic_id, parentId=parent,
            visible=owner is None, collapsed=kind == "invocation" and semantic_id in collapsed,
            hiddenOperationCount=0)
        owners[identity] = owner
        nodes.append(item)
        node_map[identity] = item
        return identity

    def port(node_id, direction, paths, kind, value_id=None, member=None):
        # At most one collapsed ancestor represents a member in a given view.
        # A view-only prefix lets the browser derive the same boundary identity
        # synchronously without hashing or changing any captured port identity.
        identity = "boundary:" + member if member is not None else _id("port", node_id, direction, paths, value_id, member)
        item = ProjectionPort(id=identity, nodeId=node_id, direction=direction, paths=paths,
            argumentKind=kind, valueId=value_id, memberPortId=member)
        if identity not in port_map:
            ports.append(item)
            port_map[identity] = item
        for path in paths:
            port_paths[node_id, tuple(path)] = identity
        return identity

    def tree_ports(node_id, field, tree, direction):
        for path, argument in _leaves(tree):
            port(node_id, direction, [[field, *path]], argument.kind,
                 argument.id if isinstance(argument, ValueRef) else None)

    input_node = node("input", capture.id)
    output_node = node("output", capture.id)
    input_paths = defaultdict(list)
    for path, argument in _leaves(walk_arguments(capture.inputs)):
        if isinstance(argument, ValueRef):
            input_paths[argument.id].append(["inputs", *path])
        else:
            port(input_node, "out", [["inputs", *path]], argument.kind)
    sources = {value: port(input_node, "out", paths, "value", value)
               for value, paths in input_paths.items()}
    if capture.output is not None:
        tree_ports(output_node, "output", walk_argument(capture.output), "in")

    call_nodes = {}
    for call in capture.invocations:
        identity = node("invocation", call.id, call_nodes.get(call.parentId))
        call_nodes[call.id] = identity
        tree_ports(identity, "arguments", walk_arguments(call.arguments), "in")
        if call.output is not None:
            tree_ports(identity, "output", walk_argument(call.output), "out")

    op_nodes = {}
    for op in capture.operations:
        identity = node("operation", op.id, call_nodes.get(op.invocationId))
        op_nodes[op.id] = identity
        tree_ports(identity, "arguments", walk_arguments(op.arguments), "in")
        if op.output is not None:
            tree_ports(identity, "output", walk_argument(op.output), "out")
        for index, effect in enumerate(op.effects):
            port(identity, "in", [["effects", index, "beforeId"]], "value", effect.beforeId)
            port(identity, "out", [["effects", index, "after"]], effect.after.kind,
                 effect.after.id if isinstance(effect.after, ValueRef) else None)

    for value in capture.values:
        producer = value.producer
        if producer is not None:
            field = "output" if producer.kind == "output" else "effects"
            sources[value.id] = port_paths[op_nodes[producer.operationId], (field, *producer.path)]
        elif value.id not in sources:
            identity = node("value", value.id, call_nodes.get(value_scopes.get(value.id)))
            sources[value.id] = port(identity, "out", [], "value", value.id)

    def displayed(port_id):
        original = port_map[port_id]
        owner = owners[original.nodeId]
        return port(owner, original.direction, [], original.argumentKind, original.valueId,
                    member=port_id) if owner is not None else port_id

    def connect(value, target, kind):
        source = sources[value]
        source_owner = owners[port_map[source].nodeId]
        target_owner = owners[port_map[target].nodeId]
        hidden = source_owner is not None and source_owner == target_owner
        edges.append(ProjectionEdge(id=_id("edge", source, target, kind), kind=kind, valueId=value,
            sourcePortId=source, targetPortId=target,
            displaySourcePortId=None if hidden else displayed(source),
            displayTargetPortId=None if hidden else displayed(target)))

    for op in capture.operations:
        identity = op_nodes[op.id]
        for path, argument in walk_arguments(op.arguments):
            if isinstance(argument, ValueRef):
                connect(argument.id, port_paths[identity, ("arguments", *path)], "argument")
        for index, effect in enumerate(op.effects):
            connect(effect.beforeId, port_paths[identity, ("effects", index, "beforeId")], "effect")
    if capture.output is not None:
        for path, argument in walk_argument(capture.output):
            if isinstance(argument, ValueRef):
                connect(argument.id, port_paths[output_node, ("output", *path)], "result")

    counts = {item.id: int(item.kind == "operation") for item in nodes}
    for item in reversed(nodes):
        if item.parentId is not None:
            counts[item.parentId] += counts[item.id]
        if item.collapsed:
            item.hiddenOperationCount = counts[item.id]
    return CaptureProjection(version=1, captureId=capture.id, sourceId=capture.source.id,
        collapsedInvocationIds=[call.id for call in capture.invocations if call.id in collapsed],
        nodes=nodes, ports=ports, edges=edges)

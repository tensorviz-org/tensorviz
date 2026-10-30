"""Map faithful captured records to the existing document boundary.

The host retains the complete capture. A live document contains bounded rendering
facts and references for paginated inspection, not another copy of the input recipe
or source/environment payload. No module is imported or executed here.
"""
from collections import defaultdict
import json

from models.architecture_models import GraphDocumentV2
from .contracts import CaptureProjection, CaptureReference, DisplayPort, IntLiteral, ProjectionPort, ValueRef
from .projection import project_capture
from .port_names import port_names
from .source_join import CaptureJoin


# The full document is retained. Compressed transport keeps its 16 MiB wire cap;
# leave room for source spans and message metadata in the 96 MiB decoded cap.
MAX_DOCUMENT_BYTES = 72 * 1024 * 1024
MAX_EDIT_CAPABILITY_BYTES = 256 * 1024


class DocumentBudgetExceeded(ValueError):
    pass


def validate_document_projection(graph):
    """Check original identities/endpoints; never execute a captured document.

    The complete capture remains the source of argument trees. This transport
    subset can prove its own ownership and port references, not replay a model.
    """
    summary = graph.metadata["capture"]
    nodes, ports, edges = [], [], []
    entities, values = set(), {}
    if summary["version"] != 1:
        raise ValueError("unsupported capture document version")
    for node in graph.nodes:
        ref = CaptureReference.model_validate(node.metadata["captureRef"])
        if ref.captureId != summary["id"] or ref.sourceId != summary["sourceId"]:
            raise ValueError("document contains a foreign capture reference")
        expected_kind = "module" if ref.kind == "invocation" else ref.kind if ref.kind in ("input", "output") else "layer"
        if node.kind != expected_kind:
            raise ValueError("document node kind disagrees with its captured reference")
        own = [DisplayPort.model_validate(port) for port in node.metadata["capturePorts"]]
        entity = (ref.kind, ref.semanticId)
        if entity in entities:
            raise ValueError("a capture entity cannot be duplicated in the document")
        entities.add(entity)
        for port in own:
            if port.valueId is not None:
                facts = (port.shape, port.dtype, port.device)
                if port.valueId in values and values[port.valueId] != facts:
                    raise ValueError("one captured value cannot have contradictory tensor facts")
                values[port.valueId] = facts
        for field, direction in (("_in_shape", "in"), ("_out_shape", "out")):
            if field in node.params:
                shape = node.params[field]
                if (not isinstance(shape, list) or len(shape) > 32
                        or any(type(dim) is not int or dim < 0 for dim in shape)
                        or not any(port.direction == direction and port.shape == shape for port in own)):
                    raise ValueError("displayed shape has no corresponding captured value")
        if any(port.nodeId != node.node_id or port.memberPortId is not None for port in own):
            raise ValueError("document must retain original member ports")
        if (node.in_ports != [port.id for port in own if port.direction == "in"]
                or node.out_ports != [port.id for port in own if port.direction == "out"]):
            raise ValueError("document port lists disagree with their records")
        nodes.append(dict(id=node.node_id, kind=ref.kind, semanticId=ref.semanticId,
            parentId=node.metadata.get("parentId"), visible=True, collapsed=False, hiddenOperationCount=0))
        ports.extend({key: value for key, value in port.model_dump().items() if key in ProjectionPort.model_fields} for port in own)
    if (sum(node["kind"] == "operation" for node in nodes) != summary["operationCount"]
            or sum(node["kind"] == "invocation" for node in nodes) != summary["invocationCount"]):
        raise ValueError("capture summary counts disagree with the retained records")
    by_port = {port["id"]: port for port in ports}
    for edge in graph.edges:
        record = edge.metadata["captureEdge"]
        if (record["id"] != edge.edge_id or record["sourcePortId"] != edge.source_port
                or record["targetPortId"] != edge.target_port
                or by_port[edge.source_port]["nodeId"] != edge.source_node_id
                or by_port[edge.target_port]["nodeId"] != edge.target_node_id):
            raise ValueError("document connection disagrees with its captured endpoints")
        edges.append(record)
    return CaptureProjection.model_validate(dict(version=1, captureId=summary["id"], sourceId=summary["sourceId"],
        collapsedInvocationIds=[], nodes=nodes, ports=ports, edges=edges))


def capture_document(payload, *, name, graph_id, model_file, source_digests, source_graph, source_spans, source_texts=None):
    joined = CaptureJoin(payload, model_file=model_file, class_name=name, source_digests=source_digests)
    capture = joined.capture
    projection = project_capture(capture)
    values = {value.id: value for value in capture.values}
    ops = {op.id: op for op in capture.operations}
    calls = {call.id: call for call in capture.invocations}
    diagnostics = {item.id: item for item in capture.diagnostics}
    modules = {module.id: module for module in capture.modules}
    definitions = {definition.id: definition for definition in capture.definitions}
    edit_resolver, navigation_resolver, edit_unavailable, edit_bytes = None, None, None, 0
    if source_texts is not None and capture.backend != 'jaxpr':
        from .source_edits import CapturedEditResolver, SourceEditRefused
        try:
            edit_resolver = CapturedEditResolver(capture, source_texts)
            navigation_resolver = edit_resolver
        except SourceEditRefused as error:
            edit_unavailable = str(error)
            if edit_unavailable == "source_limit":
                # Navigation needs only recorded call-site files. Keep the same
                # proof limits without granting edits over an incomplete source set.
                referenced = {anchor.file for binding in capture.sourceBindings for anchor in binding.anchors}
                try:
                    navigation_resolver = CapturedEditResolver(capture,
                        {path: text for path, text in source_texts.items() if path in referenced})
                except SourceEditRefused:
                    pass
    call_ordinals, module_calls = {}, defaultdict(list)
    for call in capture.invocations:
        module_calls[call.moduleId].append(call.id)
        call_ordinals[call.id] = len(module_calls[call.moduleId])
    children = defaultdict(list)
    counts = {node.id: int(node.kind == "operation") for node in projection.nodes}
    for node in reversed(projection.nodes):
        if node.parentId is not None:
            children[node.parentId].append(node.id)
            counts[node.parentId] += counts[node.id]
    port_map = {}
    node_ports = defaultdict(list)
    names = {}
    for node in projection.nodes:
        record = ops.get(node.semanticId) if node.kind == "operation" else calls.get(node.semanticId) if node.kind == "invocation" else None
        names[node.id] = (port_names(record.arguments, record.output) if record is not None else
                         port_names(capture.inputs, None, "inputs") if node.kind == "input" else
                         port_names(None, capture.output) if node.kind == "output" else {})
    for port in projection.ports:
        value = values.get(port.valueId)
        display_name = " / ".join(names[port.nodeId].get(tuple(path), "") for path in port.paths)
        display = DisplayPort(**port.model_dump(), shape=value.shape if value else None,
                              dtype=value.dtype if value else None, device=value.device if value else None,
                              displayName=display_name if display_name and len(display_name) <= 4096 else None)
        port_map[port.id] = display
        node_ports[port.nodeId].append(display)

    def shape(argument):
        return values[argument.id].shape if isinstance(argument, ValueRef) else None

    def output_metric(argument):
        value = values.get(argument.id) if isinstance(argument, ValueRef) else None
        if value is None or value.shape is None or value.logicalBytes is None:
            return {}
        return dict(outputTensor=dict(shape=value.shape, dtype=value.dtype.removeprefix("torch."),
                                      logicalBytes=value.logicalBytes))

    def boundary_shapes(arguments, output):
        inputs = list(arguments.args) + list(arguments.kwargs.values()) if arguments is not None else []
        input_shape = shape(inputs[0]) if len(inputs) == 1 else None
        output_shape = shape(output)
        return {**({"_in_shape": input_shape} if input_shape is not None else {}),
                **({"_out_shape": output_shape} if output_shape is not None else {})}

    nodes, spans, collapsed = [], {}, []
    for node in projection.nodes:
        ports = node_ports[node.id]
        ref = CaptureReference(version=1, captureId=capture.id, sourceId=capture.source.id,
                               kind=node.kind, semanticId=node.semanticId)
        metadata = dict(captureRef=ref.model_dump(mode="json"),
                        capturePorts=[port.model_dump(mode="json") for port in ports],
                        capturedOperationCount=counts[node.id])
        if node.parentId is not None:
            metadata["parentId"] = node.parentId
        if source_texts is not None and node.kind in ("operation", "invocation"):
            metadata["sourceEditCapabilities"] = []
            if edit_resolver is not None and edit_bytes < MAX_EDIT_CAPABILITY_BYTES:
                try:
                    for resolved in edit_resolver.capabilities(node.semanticId):
                        capability = resolved.model_dump(mode="json")
                        size = len(json.dumps(capability).encode())
                        if edit_bytes + size > MAX_EDIT_CAPABILITY_BYTES:
                            edit_bytes = MAX_EDIT_CAPABILITY_BYTES
                            raise SourceEditRefused("capability_limit")
                        metadata["sourceEditCapabilities"].append(capability)
                        edit_bytes += size
                except SourceEditRefused as error:
                    metadata["sourceEditLimitation"] = str(error)
            else:
                metadata["sourceEditLimitation"] = edit_unavailable or "capability_limit"
        params = {}
        op_name = label = node.kind.title()
        kind = "layer"
        status = "completed"
        if node.kind == "operation":
            op = ops[node.semanticId]
            op_name = op.target
            label = op.target
            status = op.status
            params = boundary_shapes(op.arguments, op.output)
            metadata.update(output_metric(op.output))
        elif node.kind == "invocation":
            call = calls[node.semanticId]
            module = modules[call.moduleId]
            definition = definitions[module.definitionId]
            op_name = definition.qualifiedName
            label = module.paths[0] if module.paths and module.paths[0] else definition.qualifiedName.rsplit(".", 1)[-1]
            if len(module.paths) > 1:
                label += " · shared instance"
            if len(module_calls[call.moduleId]) > 1:
                label += f" · call {call_ordinals[call.id]}/{len(module_calls[call.moduleId])}"
            params = boundary_shapes(call.arguments, call.output)
            metadata.update(output_metric(call.output))
            if call.parameterCount is not None:
                metadata["parameterCount"] = call.parameterCount
            status = call.status
            kind = "module"
            metadata.update(isGroup=True, moduleInstanceId=node.id,
                moduleTemplateId=definition.qualifiedName.rsplit(".", 1)[-1],
                moduleMembers=list(reversed(children[node.id])), moduleLayerCount=len(children[node.id]),
                expandable=bool(children[node.id]))
            if call.id != capture.rootInvocationId:
                collapsed.append(node.id)
            # Registered-instance correspondence permits navigation to a unique
            # declaration, not a promise that one observed call is independently editable.
            candidates = [source_spans[path] for path in module.paths if path in source_spans]
            if candidates and len(candidates) == len(module.paths) and all(span == candidates[0] for span in candidates):
                spans[node.id] = dict(candidates[0])
                metadata["declarationSpan"] = dict(candidates[0])
        elif node.kind == "value":
            value = values[node.semanticId]
            op_name = "State" if value.state else "External value"
            label = value.state[0].name if value.state else "External value"
            params = {"_out_shape": value.shape} if value.shape is not None else {}
            metadata.update(output_metric(ValueRef(kind="value", id=value.id)))
            if value.state:
                candidates = [source_spans[reference.name] for reference in value.state if reference.name in source_spans]
                # Shared names must agree on one declaration. A capture label
                # alone never proves which source assignment created the state.
                if candidates and len(candidates) == len(value.state) and all(span == candidates[0] for span in candidates):
                    metadata["sourceKind"] = "parameter"
                    spans[node.id] = dict(candidates[0])
        else:
            kind = node.kind
            op_name = "Input" if kind == "input" else "Output"
            label = "Inputs" if kind == "input" else "Outputs"
            # Multi-tensor boundaries live in their typed ports, never a first tensor shape.
            if len(ports) == 1 and ports[0].shape is not None:
                params["_out_shape" if kind == "input" else "_in_shape"] = ports[0].shape
            if kind == "input":
                inputs = [*capture.inputs.args, *capture.inputs.kwargs.values()]
                if len(inputs) == 1:
                    metadata.update(output_metric(inputs[0]))
            if kind == "output" and capture.output is None:
                status = "interrupted"
        params["_shape_provenance"] = "observed" if status == "completed" else "failed" if status == "failed" else "unknown"
        if capture.backend == 'jaxpr' and status == 'completed':
            params['_shape_provenance'] = 'inferred'
        if navigation_resolver is not None and node.kind in ("operation", "invocation"):
            record = ops[node.semanticId] if node.kind == "operation" else calls[node.semanticId]
            span = navigation_resolver.navigation(record.sourceBindingIds)
            if span is not None:
                spans[node.id] = span
                if (node.kind == "operation" and record.target == "aten.slice.Tensor"
                        and len(record.arguments.args) > 1 and isinstance(record.arguments.args[1], IntLiteral)):
                    metadata["captureSourceExpression"] = record.sourceBindingIds[0]
                    metadata["captureSliceAxis"] = int(record.arguments.args[1].value)
        if capture.backend == 'jaxpr' and node.kind in ('operation', 'invocation'):
            record = ops[node.semanticId] if node.kind == 'operation' else calls[node.semanticId]
            bindings = [binding for binding in capture.sourceBindings if binding.id in record.sourceBindingIds]
            if len(bindings) == 1 and len(bindings[0].anchors) == 1:
                anchor = bindings[0].anchors[0]
                spans[node.id] = dict(file=anchor.file, start=[anchor.startLine, anchor.startColumn],
                    end=[anchor.endLine, anchor.endColumn], kind='call', digest=anchor.digest)
        metadata.update(captureLabel=label, shapeEvidence=("inferred" if capture.backend == 'jaxpr' else "measured") if status == "completed" else "unknown",
            layerType=dict(id="module" if kind == "module" else kind if kind in ("input", "output") else "captured_operation",
                           name=op_name, displayName=op_name.rsplit(".", 1)[-1] if kind == "module" else op_name,
                           category="module" if kind == "module" else "io" if kind in ("input", "output") else "misc",
                           icon="Package" if kind == "module" else "Box", description="JAX traced program" if capture.backend == 'jaxpr' else "Captured execution", parameters=[]))
        if status == "failed":
            metadata["error"] = "This captured call failed. Inspect its recorded inputs and diagnostic."
            record = ops[node.semanticId] if node.kind == "operation" else calls[node.semanticId]
            diagnostic = diagnostics.get(record.diagnosticId)
            if diagnostic is not None:
                metadata["captureDiagnostic"] = diagnostic.model_dump(mode="json")
        nodes.append(dict(node_id=node.id, kind=kind, op=op_name, params=params,
            in_ports=[port.id for port in ports if port.direction == "in"],
            out_ports=[port.id for port in ports if port.direction == "out"], metadata=metadata))

    edges = [dict(edge_id=edge.id, source_node_id=port_map[edge.sourcePortId].nodeId, source_port=edge.sourcePortId,
                  target_node_id=port_map[edge.targetPortId].nodeId, target_port=edge.targetPortId,
                  metadata=dict(captureEdge=edge.model_dump(mode="json"))) for edge in projection.edges]
    source_coverage = [joined.correspondence(node).model_dump(mode="json") for node in source_graph.get("nodes", [])
                       if node.get("id") not in ("input", "output")]
    # Conditional/factory declarations may have no lowered source card. Keep the
    # recorded instance paths so an exact source review can still locate their calls.
    mapped_paths = {fact["nodeId"] for fact in source_coverage}
    source_coverage.extend(joined.correspondence({"id": path}).model_dump(mode="json")
                           for path in joined.modules if path and path not in mapped_paths)
    root = calls.get(capture.rootInvocationId)
    failure_id = None
    if root is not None and root.status == "failed":
        failing_op = next((op for op in reversed(capture.operations)
                           if op.status == "failed" and op.diagnosticId == root.diagnosticId), None)
        failure_id = next(node.id for node in projection.nodes
                          if node.kind == ("operation" if failing_op else "invocation")
                          and node.semanticId == (failing_op.id if failing_op else root.id))
    document = GraphDocumentV2.model_validate(dict(schemaVersion=2,
        semantic=dict(graph_id=graph_id, name=name, revision=0, inputs=[], nodes=nodes, edges=edges, modules=[], operations=[],
            metadata=dict(capture=dict(version=1, id=capture.id, sourceId=capture.source.id, status=capture.status,
                backend=capture.backend, rootInvocationId=capture.rootInvocationId, omissions=capture.coverage.omissions,
                operationCount=len(capture.operations), invocationCount=len(capture.invocations), failureNodeId=failure_id),
                sourceCorrespondence=source_coverage)),
        view=dict(graph_id=graph_id, node_positions={node["node_id"]: dict(x=0, y=0) for node in nodes},
                  collapsed_modules=collapsed, selection=[], layout=dict(strategy="captured"))))
    if source_texts is not None:
        from services.ide.review_declarations import review_declarations, captured_declaration_calls
        evidence = review_declarations(source_texts)
        document.semantic.metadata["reviewDeclarations"] = evidence
        document.semantic.metadata["sourceDeclarationCalls"] = captured_declaration_calls(capture, evidence, navigation_resolver)
    if len(document.model_dump_json().encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise DocumentBudgetExceeded("captured document exceeds the display transport budget; the host still retains the capture")
    return document, spans

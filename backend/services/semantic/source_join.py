"""Join saved source cards to captured module/state identities without executing code.

Module paths identify instances in one exact snapshot. They do not identify a
specific call expression or specialize an alias of a shared module.
"""
from collections import defaultdict
from copy import deepcopy

from .contracts import CaptureSnapshot, SourceCorrespondence


class CaptureJoin:
    def __init__(self, payload, *, model_file, class_name, source_digests):
        self.capture = CaptureSnapshot.model_validate(payload)
        source = self.capture.source
        files = {item.path: item.digest for item in source.files if item.role == "source"}
        if (source.modelFile != model_file or source.className != class_name
                or source_digests.get(model_file) != files.get(model_file)
                or any(files.get(file) != digest for file, digest in source_digests.items())):
            raise ValueError("capture and projected source snapshots differ")
        self.modules = {path: module for module in self.capture.modules for path in module.paths}
        self.calls = defaultdict(list)
        for call in self.capture.invocations:
            self.calls[call.moduleId].append(call)
        self.values = {value.id: value for value in self.capture.values}
        self.state = defaultdict(list)
        for value in self.capture.values:
            for reference in value.state:
                self.state[reference.name].append(value.id)

    def correspondence(self, node):
        data = node.get("data") or {}
        params = data.get("parameters") or {}
        node_id = str(node["id"])
        path = params.get("_module_path") or node_id
        op = str((data.get("layerType") or {}).get("id") or "").rsplit(".", 1)[-1].lower()
        base = dict(nodeId=node_id, captureId=self.capture.id, sourceId=self.capture.source.id,
                    moduleId=None, invocationIds=[], stateValueIds=[], reason=None)
        if op in ("parameter", "buffer", "embeddingbag_weight") or path in self.state:
            base["stateValueIds"] = list(self.state[path])
            return SourceCorrespondence(**base, status="state", basis="state_reference")
        module = self.modules.get(path)
        omissions = self.capture.coverage.omissions
        if module is not None:
            calls = self.calls[module.id]
            status = ("failed" if any(call.status == "failed" for call in calls) else "observed") if calls else (
                "capture_limited" if omissions else "not_observed")
            base.update(moduleId=module.id, invocationIds=[call.id for call in calls],
                        reason=omissions[0] if status == "capture_limited" else None)
            return SourceCorrespondence(**base, status=status,
                basis="shared_instance" if len(module.paths) > 1 else "registered_path")
        is_module = params.get("_kind") == "class" or bool(params.get("_module_path"))
        status = "capture_limited" if is_module and omissions else "static_only" if is_module else "unresolved"
        base["reason"] = omissions[0] if status == "capture_limited" else None
        return SourceCorrespondence(**base, status=status, basis="no_mapping")

    def _shape(self, argument):
        if argument is None or argument.kind != "value":
            return None
        value = self.values[argument.id]
        if value.shape is None or not all(type(dim) is int and dim >= 0 for dim in value.shape):
            return None
        return value.shape, value.dtype

    def _same_shape(self, arguments):
        shapes = [self._shape(argument) for argument in arguments]
        if not shapes or shapes[0] is None or any(shape != shapes[0] for shape in shapes):
            return None
        return list(shapes[0][0])

    def decorate(self, node):
        """Only summarize one tensor with the same measured shape in every call.

        Ordered/multiple outputs and calls with different shapes remain in capture;
        the legacy single-shape card cannot summarize them by taking the first.
        """
        node = deepcopy(node)
        data = node.setdefault("data", {})
        params = data.setdefault("parameters", {})
        joined = self.correspondence(node)
        params["_capture_coverage"] = joined.model_dump(mode="json")
        output_tensor = data.get("outputTensor")
        for key in ("_in_shape", "_out_shape", "_declared_out_shape", "_call_sites"):
            params.pop(key, None)
        # Cached first-tensor metrics cannot fill a missing canonical summary.
        for key in ("inputShape", "outputShape", "outputTensor", "shapeEvidence"):
            data.pop(key, None)
        params["_shape_provenance"] = joined.status
        calls = self.calls[joined.moduleId] if joined.moduleId is not None else []
        if calls:
            inputs = [list(call.arguments.args) + list(call.arguments.kwargs.values()) for call in calls]
            input_shape = self._same_shape([items[0] if len(items) == 1 else None for items in inputs])
            output_shape = self._same_shape([call.output for call in calls])
            for key, shape in (("_in_shape", input_shape), ("_out_shape", output_shape)):
                if shape is not None:
                    params[key] = "x".join(map(str, shape)) if shape else []
            if len(calls) > 1:
                params["_call_sites"] = len(calls)
            if (output_shape is not None and isinstance(output_tensor, dict)
                    and output_tensor.get("shape") == output_shape
                    and output_tensor.get("dtype") == (self._shape(calls[0].output)[1] or "").removeprefix("torch.")):
                data["outputTensor"] = output_tensor
        if joined.status == "state":
            shapes = [(self.values[id].shape, self.values[id].dtype) for id in joined.stateValueIds]
            if shapes and all(shape == shapes[0] for shape in shapes) and shapes[0][0] is not None:
                shape = shapes[0][0]
                if all(type(dim) is int and dim >= 0 for dim in shape):
                    params["_out_shape"] = "x".join(map(str, shape)) if shape else []
        return node

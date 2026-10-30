"""Regenerate small CPU dimension-rule oracles from actual PyTorch operations.

Only these fixed test cases execute. No workspace model, checkpoint or agent is
loaded. The JSON is committed so TypeScript tests do not require Python/Torch.
"""
import argparse
import hashlib
from fixture_json import dumps
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


class WidthPath(torch.nn.Module):
    def __init__(self, width=12, heads=3):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(width))
        self.heads = heads

    def forward(self, inputs):
        projected = inputs @ self.weight
        split = projected.reshape(inputs.shape[0], self.heads, -1)
        return split + split


class LayerNormPath(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.bias = torch.nn.Parameter(torch.zeros(4))

    def forward(self, inputs):
        output, mean, rstd = torch.ops.aten.native_layer_norm.default(inputs, [4], self.weight, self.bias, 1e-5)
        return output + mean, rstd + mean


@dataclass
class Tensor:
    shape: tuple


def build():
    cases = []

    def record(target, *arguments, **keywords):
        inputs = {}

        def convert(value):
            if isinstance(value, Tensor):
                name = f"input:{len(inputs)}"
                inputs[name] = list(value.shape)
                return torch.ones(value.shape), dict(kind="value", id=name)
            if isinstance(value, int):
                return value, dict(kind="int", value=str(value))
            if isinstance(value, float):
                return value, dict(kind="float", value=str(value))
            if value is None:
                return None, dict(kind="none")
            if isinstance(value, (tuple, list)):
                pairs = [convert(item) for item in value]
                return [pair[0] for pair in pairs], dict(kind="list", items=[pair[1] for pair in pairs])
            raise TypeError(type(value))

        args, kwargs = [convert(value) for value in arguments], {key: convert(value) for key, value in keywords.items()}
        _, name, overload = target.split(".")
        operation = getattr(getattr(torch.ops.aten, name), overload)
        try:
            output = operation(*[pair[0] for pair in args], **{key: pair[0] for key, pair in kwargs.items()})
            if isinstance(output, tuple):
                assert all(isinstance(value, torch.Tensor) for value in output)
                expected = dict(status="compatible", shape=None, shapes=[list(value.shape) for value in output])
            else:
                assert isinstance(output, torch.Tensor)
                expected = dict(status="compatible", shape=list(output.shape))
        except (RuntimeError, IndexError) as error:
            expected = dict(status="incompatible", shape=None, errorType=type(error).__name__)
        cases.append(dict(id=f"{target}:{len(cases)}", target=target, inputs=inputs,
            arguments=dict(args=[pair[1] for pair in args], kwargs={key: pair[1] for key, pair in kwargs.items()}),
            expected=expected))

    shapes = [(), (4,), (1, 4), (2, 4), (0, 4), (2, 0, 4), (3,)]
    for name in ("add", "sub", "mul", "div"):
        for left in shapes:
            for right in shapes:
                record(f"aten.{name}.Tensor", Tensor(left), Tensor(right))
        record(f"aten.{name}.Scalar", Tensor((2, 4)), 2)
    for name in ("relu", "gelu", "sigmoid", "tanh", "neg", "clone", "detach"):
        for shape in ((), (0, 4), (2, 3, 4)):
            record(f"aten.{name}.default", Tensor(shape))
    for left, right in [((2, 3), (3, 4)), ((0, 3), (3, 4)), ((2, 0), (0, 4)),
                        ((2, 3), (4, 2)), ((3,), (3, 4)), ((2, 3), (3,))]:
        record("aten.mm.default", Tensor(left), Tensor(right))
    for left, right in [((2, 3, 4), (2, 4, 5)), ((0, 3, 4), (0, 4, 5)),
                        ((1, 3, 4), (2, 4, 5)), ((2, 3, 4), (2, 5, 4)), ((3, 4), (4, 5))]:
        record("aten.bmm.default", Tensor(left), Tensor(right))
    for bias in [(), (4,), (1, 4), (2, 4), (2, 1), (3,), (1, 2, 4)]:
        record("aten.addmm.default", Tensor(bias), Tensor((2, 3)), Tensor((3, 4)))
        record("aten.linear.default", Tensor((2, 3)), Tensor((4, 3)), Tensor(bias))
    record("aten.linear.default", Tensor((2, 3, 5)), Tensor((4, 5)), None)
    record("aten.linear.default", Tensor((2, 3)), Tensor((4, 5)), None)
    for name in ("view", "reshape", "_unsafe_view"):
        for shape, dims in [((2, 12), (2, 3, 4)), ((2, 12), (2, 5, -1)), ((2, 12), (2, 3, -1)),
                            ((2, 12), (-1, -1)), ((0, 12), (0, -1)), ((0, 12), (-1, 12)),
                            ((0, 12), (0, 4)), ((), (1,)), ((1,), ()), ((2, 4), (-2, 4))]:
            record(f"aten.{name}.default", Tensor(shape), dims)
    for shape, dims in [((2, 3, 4), (2, 0, 1)), ((2, 3, 4), (-1, 0, 1)), ((2, 3), (0, 0)),
                        ((2, 3), (0, 2)), ((2, 3), (0,)), ((), ())]:
        record("aten.permute.default", Tensor(shape), dims)
    for shape in ((), (3,), (2, 3), (2, 3, 4)):
        record("aten.t.default", Tensor(shape))
        for dims in [(0, -1), (1, 0), (0, 3)]:
            record("aten.transpose.int", Tensor(shape), *dims)
    for shapes_, dim in [([(2, 3), (4, 3)], 0), ([(2, 3), (2, 4)], -1), ([(2, 3), (4, 3)], 1),
                         ([(0,), (2, 3)], 1), ([(0,), (0,)], 0), ([(0,), (0,)], 1),
                         ([(0,), (0,)], -100), ([(0,), (0,)], 100),
                         ([(), ()], 0), ([(2, 3), (2, 3)], 2), ([], 0)]:
        record("aten.cat.default", [Tensor(shape) for shape in shapes_], dim)
    record("aten.mm.default", self=Tensor((2, 3)), mat2=Tensor((3, 4)))
    for shape, normalized in [((2, 4), (4,)), ((2, 3, 4), (3, 4)), ((4,), (4,)),
                               ((0, 3, 4), (3, 4)), ((2, 0), (0,)), ((0,), (0,)),
                               ((2, 3, 4), (4,)), ((2, 4), (3,)), ((2, 4), (2, 4)),
                               ((2, 4), (1, 2, 4)), ((), ()), ((2, 4), ()),
                               ((2, 4), (-1,)), ((2, 4), (0,))]:
        for weight, bias in [(None, None), (Tensor(normalized), None),
                             (None, Tensor(normalized)), (Tensor(normalized), Tensor(normalized))]:
            if any(dim < 0 for dim in normalized) and (weight or bias):
                continue
            record("aten.native_layer_norm.default", Tensor(shape), normalized, weight, bias, 1e-5)
    for weight, bias in [(Tensor((1, 4)), None), (None, Tensor((1, 4))),
                         (Tensor((3,)), Tensor((4,))), (Tensor((4,)), Tensor((3,)))]:
        record("aten.native_layer_norm.default", Tensor((2, 4)), [4], weight, bias, 1e-5)
    record("aten.native_layer_norm.default", input=Tensor((2, 4)), normalized_shape=[4],
        weight=Tensor((4,)), bias=Tensor((4,)), eps=1e-5)
    return dict(version=1, python=platform.python_version(), torch=torch.__version__,
                scope="CPU float32 dense dimension checks only; contiguous tensors", cases=cases)


def width_capture(*, layer_norm=False):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "backend"))
    from services.ingest.trace import trace_model
    from services.semantic.contracts import CaptureLimits, CaptureSnapshot, SourceSnapshot
    from services.semantic.recorder import CaptureContext
    source = Path(__file__).read_bytes()
    name = "layer-norm" if layer_norm else "width"
    context = CaptureContext(source=SourceSnapshot(version=1, id=f"source:{name}-oracle",
        workspaceSessionId="session:oracle", workspaceId="workspace:oracle", modelId=f"model:{name}",
        modelFile="tooling/ml_shape_oracles.py", className="LayerNormPath" if layer_norm else "WidthPath", generation=0,
        files=[dict(path="tooling/ml_shape_oracles.py", digest=hashlib.sha256(source).hexdigest(), documentVersion=None, role="source")],
        dependencyScope="partial", configDigest=None, environmentDigest="b" * 64, sourceMode="saved"),
        plan_digest="c" * 64, attempt_id=f"attempt:{name}-oracle",
        limits=CaptureLimits(operations=100, invocations=100, values=100, bytes=1024 * 1024))
    model = LayerNormPath() if layer_norm else WidthPath()
    result = trace_model(model, (torch.ones(2, 4 if layer_norm else 12),), tier="real", semantic_context=context, record_evidence=True)
    assert result.error is None, result.error
    capture = CaptureSnapshot.model_validate(result.semantic_capture)
    assert capture.status == "complete"
    candidates = []
    for width in (() if layer_norm else (12, 15, 16)):
        try:
            output = WidthPath(width)(torch.ones(2, width))
            candidates.append(dict(width=width, status="passed", shape=list(output.shape)))
        except RuntimeError:
            candidates.append(dict(width=width, status="failed", shape=None))
    if layer_norm:
        for shape in [(2, 4), (3, 4), (0, 4), (3, 5)]:
            try:
                output = LayerNormPath()(torch.ones(shape))
                candidates.append(dict(inputShape=list(shape), status="passed", shapes=[list(value.shape) for value in output]))
            except RuntimeError:
                candidates.append(dict(inputShape=list(shape), status="failed", shapes=None))
    return dict(version=1, python=platform.python_version(), torch=torch.__version__,
        scope="Real CPU dispatch capture with a test-owned context; candidate forwards are independent shape oracles, not native review acceptance",
        capture=capture.model_dump(mode="json"), candidates=candidates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--capture-output", type=Path)
    parser.add_argument("--layer-norm-capture-output", type=Path)
    options = parser.parse_args()
    result = build()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(dumps(result) + "\n")
    passed = sum(case["expected"]["status"] == "compatible" for case in result["cases"])
    print(f"Wrote {len(result['cases'])} actual PyTorch cases: {passed} compatible, {len(result['cases']) - passed} incompatible")
    if options.capture_output:
        capture = width_capture()
        options.capture_output.parent.mkdir(parents=True, exist_ok=True)
        options.capture_output.write_text(dumps(capture) + "\n")
        print(f"Wrote real width-change capture and {len(capture['candidates'])} independent candidate forwards")
    if options.layer_norm_capture_output:
        capture = width_capture(layer_norm=True)
        options.layer_norm_capture_output.parent.mkdir(parents=True, exist_ok=True)
        options.layer_norm_capture_output.write_text(dumps(capture) + "\n")
        print(f"Wrote real LayerNorm capture and {len(capture['candidates'])} independent candidate forwards")

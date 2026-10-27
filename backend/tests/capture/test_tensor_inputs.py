"""MCP input bounds and actual typed worker execution, without evaluating input text."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from services.ingest.tensor_inputs import validate_tensor_inputs, make_tensor_inputs

BACKEND = Path(__file__).parent.resolve().parent.parent
BASE = {"shape": [1, 4], "dtype": "int64", "fill": "ones"}

@pytest.mark.parametrize("value", [[], "torch.ones(2)", [{**BASE, "shape": [True]}],
    [{**BASE, "shape": [0]}], [{**BASE, "shape": [1.5]}], [{**BASE, "device": "cuda"}],
    [{**BASE, "fill": "randn"}], [{**BASE, "dtype": []}], [{**BASE, "shape": [1000, 5000]}],
    [BASE] * 9, [{**BASE, "shape": [1] * 9}]])
def test_refuses_invalid_descriptions(value):
    with pytest.raises(ValueError):
        validate_tensor_inputs(value)


def test_input_factory_retains_dtype_and_deterministic_values():
    import torch
    specs = [BASE, {"shape": [2, 3], "dtype": "float32", "fill": "randn"}]
    a, kwargs = make_tensor_inputs(specs)
    b, _ = make_tensor_inputs(specs)
    assert kwargs == {} and a[0].dtype == torch.int64
    assert a[0].tolist() == [[1, 1, 1, 1]] and torch.equal(a[1], b[1])


def test_bad_input_exits_before_torch_or_model_import(tmp_path):
    marker = tmp_path / "executed"
    model = tmp_path / "model.py"
    model.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    process = subprocess.run([sys.executable, "-S", "-m", "services.ingest.worker",
        "--file", str(model), "--cls", "Net", "--tensor-inputs", json.dumps([{**BASE, "shape": [1000, 5000]}])],
        cwd=BACKEND, capture_output=True, text=True, timeout=30)
    assert process.returncode == 2 and json.loads(process.stdout)["error"]["type"] == "UsageError"
    assert not marker.exists()


def test_worker_uses_positional_integer_inputs_instead_of_source_input_expression(tmp_path):
    model = tmp_path / "model.py"
    model.write_text("""import torch
from torch import nn
# tensorviz: input=(_ for _ in ()).throw(RuntimeError('must not evaluate'))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 3)
    def forward(self, ids, mask):
        assert ids.dtype == torch.int64 and mask.dtype == torch.float32
        return self.embedding(ids) * mask
""")
    inputs = [BASE, {"shape": [1, 4, 3], "dtype": "float32", "fill": "ones"}]
    process = subprocess.run([sys.executable, "-m", "services.ingest.worker", "--file", str(model),
        "--cls", "Net", "--tensor-inputs", json.dumps(inputs)], cwd=BACKEND,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}, capture_output=True, text=True, timeout=60)
    envelope = json.loads(process.stdout)
    assert process.returncode == 0, envelope.get("error")
    assert envelope["error"] is None and envelope["result"]["graph"]["nodes"]
    assert envelope["result"]["diagnostics"]["example_input"] == [[1, 4], [1, 4, 3]]

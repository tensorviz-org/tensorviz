"""QA regression suite — export round-trip & code generation (PRs #32, #46).

The product promise for the export path (verification_service.export_package +
code_generation.CodeGenerationService) is that a downloaded package is
*independently runnable*: `python sample_run.py` in the unzipped folder must run
and print the SAME output shape that verify_model reports for the same graph.

These tests exercise that promise end-to-end for the three staple architecture
families a user builds on the canvas — a flat MLP, a CNN (conv → flatten →
linear) and a recurrent LSTM stack — plus a graph with a *reused* block, which
must export as one class instantiated ×N via nn.ModuleList (PR #46 modular
codegen) rather than N inlined copies. Unlike test_export_roundtrip.py (which
execs the generated model.py IN-PROCESS on gallery/registry templates), here we
run the shipped `sample_run.py` in a real SUBPROCESS from the extracted package
dir — the exact `python sample_run.py` a user runs — so the "self-contained,
independently runnable" contract is what is actually verified.

Two CONFIRMED bugs are pinned with xfail(strict=False) so the suite stays green
and each auto-flips to XPASS when fixed:
  * BUG (verification_service.py:708) — sample_run.py omits model.eval(), so a
    BatchNorm model at batch=1 crashes even though verify_model and model.py's
    own __main__ (which DOES call eval()) both succeed.
  * BUG (verification_service.py:143) — verify_model seeds Embedding inputs with
    a hardcoded randint(0, 100), so a valid small-vocab embedding graph is
    false-rejected even though its exported code (which uses
    randint(0, num_embeddings)) runs fine.

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest \
        tests/test_qa_export_roundtrip.py -q -p no:cacheprovider
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import zipfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede torch import
os.environ.setdefault("AUTH_DISABLED", "true")


from services.verification_service import verify_model, export_package

MODEL_NAME = "RoundtripModel"
# The suite runs under the viz interpreter (KMP_DUPLICATE_LIB_OK set above), so
# reusing it for the child guarantees torch imports identically in sample_run.py.
PY = sys.executable


# ── graph builders (flat canvas payloads, the shape verify/export consume) ────
def _node(nid, ntype, lt_id, lt_name, params):
    """A flat canvas node. layerType.name is the bare torch class ("Linear",
    "Conv2d", …) so codegen emits nn.<name>, while layerType.id is the lowercase
    palette id verify_model maps to the same class — both paths agree."""
    return {
        "id": nid,
        "type": ntype,
        "position": {"x": 0.0, "y": 0.0},
        "data": {
            "layerType": {"id": lt_id, "name": lt_name, "parameters": []},
            "parameters": params,
        },
    }


def _mlp():
    """Input(1,64) → Linear(64,32) → ReLU → Linear(32,10) → Output ⇒ (1,10)."""
    return {
        "nodes": [
            _node("input", "inputNode", "input", "Input", {"shape": [1, 64]}),
            _node("l1", "layerNode", "linear", "Linear", {"in_features": 64, "out_features": 32}),
            _node("r1", "layerNode", "relu", "ReLU", {}),
            _node("l2", "layerNode", "linear", "Linear", {"in_features": 32, "out_features": 10}),
            _node("output", "outputNode", "output", "Output", {}),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "l1"},
            {"id": "e2", "source": "l1", "target": "r1"},
            {"id": "e3", "source": "r1", "target": "l2"},
            {"id": "e4", "source": "l2", "target": "output"},
        ],
    }, [1, 64]


def _cnn():
    """Input(1,3,8,8) → Conv2d(3,4,k3,p1) → ReLU → Flatten → Linear(256,10) ⇒ (1,10)."""
    return {
        "nodes": [
            _node("input", "inputNode", "input", "Input", {"shape": [1, 3, 8, 8]}),
            _node("c1", "layerNode", "conv2d", "Conv2d",
                  {"in_channels": 3, "out_channels": 4, "kernel_size": 3, "padding": 1}),
            _node("r1", "layerNode", "relu", "ReLU", {}),
            _node("f1", "layerNode", "flatten", "Flatten", {}),
            _node("l1", "layerNode", "linear", "Linear", {"in_features": 256, "out_features": 10}),
            _node("output", "outputNode", "output", "Output", {}),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "c1"},
            {"id": "e2", "source": "c1", "target": "r1"},
            {"id": "e3", "source": "r1", "target": "f1"},
            {"id": "e4", "source": "f1", "target": "l1"},
            {"id": "e5", "source": "l1", "target": "output"},
        ],
    }, [1, 3, 8, 8]


def _rnn():
    """Input(1,12,8) → LSTM(8→16, batch_first) → Linear(16,4) → Output ⇒ (1,12,4).

    Exercises the recurrent-op codegen: the LSTM returns (output, (h,c)), which
    codegen unwraps as `self.lstm(x)[0]` so the next Linear gets a tensor."""
    return {
        "nodes": [
            _node("input", "inputNode", "input", "Input", {"shape": [1, 12, 8]}),
            _node("lstm", "layerNode", "lstm", "LSTM",
                  {"input_size": 8, "hidden_size": 16, "batch_first": True}),
            _node("l1", "layerNode", "linear", "Linear", {"in_features": 16, "out_features": 4}),
            _node("output", "outputNode", "output", "Output", {}),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "lstm"},
            {"id": "e2", "source": "lstm", "target": "l1"},
            {"id": "e3", "source": "l1", "target": "output"},
        ],
    }, [1, 12, 8]


def _reused(repeat: int = 4):
    """Input(1,16) → [Linear(16,16) → ReLU] ×`repeat` → Linear(16,4) → Output.

    The repeated 2-node block must be factored by the modularizer into ONE
    reusable class instantiated ×`repeat` via nn.ModuleList (PR #46)."""
    nodes = [_node("input", "inputNode", "input", "Input", {"shape": [1, 16]})]
    edges = []
    prev = "input"
    for i in range(repeat):
        ln, rn = f"l{i}", f"r{i}"
        nodes.append(_node(ln, "layerNode", "linear", "Linear", {"in_features": 16, "out_features": 16}))
        nodes.append(_node(rn, "layerNode", "relu", "ReLU", {}))
        edges.append({"id": f"e_{prev}_{ln}", "source": prev, "target": ln})
        edges.append({"id": f"e_{ln}_{rn}", "source": ln, "target": rn})
        prev = rn
    nodes.append(_node("head", "layerNode", "linear", "Linear", {"in_features": 16, "out_features": 4}))
    nodes.append(_node("output", "outputNode", "output", "Output", {}))
    edges.append({"id": "e_head", "source": prev, "target": "head"})
    edges.append({"id": "e_out", "source": "head", "target": "output"})
    return {"nodes": nodes, "edges": edges}, [1, 16]


def _mlp_batchnorm_batch1():
    """Flat MLP with BatchNorm1d at batch=1 (the BUG-1 repro).
    Input(1,64) → Linear(64,32) → BatchNorm1d(32) → ReLU → Linear(32,10) → (1,10)."""
    return {
        "nodes": [
            _node("input", "inputNode", "input", "Input", {"shape": [1, 64]}),
            _node("l1", "layerNode", "linear", "Linear", {"in_features": 64, "out_features": 32}),
            _node("bn", "layerNode", "batchnorm1d", "BatchNorm1d", {"num_features": 32}),
            _node("r1", "layerNode", "relu", "ReLU", {}),
            _node("l2", "layerNode", "linear", "Linear", {"in_features": 32, "out_features": 10}),
            _node("output", "outputNode", "output", "Output", {}),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "l1"},
            {"id": "e2", "source": "l1", "target": "bn"},
            {"id": "e3", "source": "bn", "target": "r1"},
            {"id": "e4", "source": "r1", "target": "l2"},
            {"id": "e5", "source": "l2", "target": "output"},
        ],
    }, [1, 64]


def _embedding(num_embeddings: int):
    """Input(2,8) → Embedding(num_embeddings, 8) → Linear(8,4) → Output ⇒ (2,8,4)."""
    return {
        "nodes": [
            _node("input", "inputNode", "input", "Input", {"shape": [2, 8]}),
            _node("emb", "layerNode", "embedding", "Embedding",
                  {"num_embeddings": num_embeddings, "embedding_dim": 8}),
            _node("l1", "layerNode", "linear", "Linear", {"in_features": 8, "out_features": 4}),
            _node("output", "outputNode", "output", "Output", {}),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "emb"},
            {"id": "e2", "source": "emb", "target": "l1"},
            {"id": "e3", "source": "l1", "target": "output"},
        ],
    }, [2, 8]


# ── export / run helpers ──────────────────────────────────────────────────────
def _export_and_extract(payload, shape, dest, name=MODEL_NAME):
    """Run the product export exactly as /api/models/export-package does, unzip
    into `dest`, and return (package_dir, model_py_text, sample_run_py_text)."""
    zip_bytes, _ = export_package(payload, shape, name)
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    zf.extractall(dest)
    pkg_dir = os.path.join(dest, name)
    model_py = zf.read(f"{name}/model.py").decode("utf-8")
    sample_py = zf.read(f"{name}/sample_run.py").decode("utf-8")
    return pkg_dir, model_py, sample_py


def _run(pkg_dir, script):
    """Run one script from the extracted package dir in a fresh subprocess — the
    real `python <script>` a user runs. torch import is slow, hence the timeout."""
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return subprocess.run(
        [PY, script], cwd=pkg_dir, env=env,
        capture_output=True, text=True, timeout=300,
    )


def _printed_shape(stdout):
    """Parse the '(1, 10)' tuple the generated scripts print (both sample_run.py's
    'output shape:' and model.py __main__'s 'Output shape:') into a list[int]."""
    m = re.search(r"shape:\s*\(([^)]*)\)", stdout)
    assert m, f"no output shape line found in stdout:\n{stdout}"
    return [int(v) for v in m.group(1).split(",") if v.strip()]


def _assert_round_trips(payload, shape, tmp_path, expected):
    """The core parity gate: verify_model's shape == the shape the exported
    sample_run.py actually prints when executed as an independent script."""
    v = verify_model(payload, shape)
    assert v["ok"], f"verify_model failed at stage {v.get('stage')}: {v.get('errors')}"
    # Pin the concrete value so verify+codegen can't silently drift together.
    assert v["output_shape"] == expected, v["output_shape"]

    pkg_dir, _model_py, _sample_py = _export_and_extract(payload, shape, str(tmp_path))
    proc = _run(pkg_dir, "sample_run.py")
    assert proc.returncode == 0, f"`python sample_run.py` crashed:\n{proc.stderr}"
    assert _printed_shape(proc.stdout) == v["output_shape"], (
        f"exported code printed {proc.stdout!r} != verified {v['output_shape']}"
    )


# ── core round-trip parity (MLP / CNN / RNN) ──────────────────────────────────
def test_mlp_sample_run_reproduces_verified_shape(tmp_path):
    _assert_round_trips(*_mlp(), tmp_path=tmp_path, expected=[1, 10])


def test_cnn_sample_run_reproduces_verified_shape(tmp_path):
    _assert_round_trips(*_cnn(), tmp_path=tmp_path, expected=[1, 10])


def test_rnn_lstm_sample_run_reproduces_verified_shape(tmp_path):
    _assert_round_trips(*_rnn(), tmp_path=tmp_path, expected=[1, 12, 4])


# ── modular codegen: a reused block exports ×N, not N copies (PR #46) ──────────
def test_reused_block_exports_modulelist_and_round_trips(tmp_path):
    payload, shape = _reused(repeat=4)
    v = verify_model(payload, shape)
    assert v["ok"], v.get("errors")
    assert v["output_shape"] == [1, 4]

    pkg_dir, model_py, _sample_py = _export_and_extract(payload, shape, str(tmp_path))

    # Factored into ONE reusable block instantiated 4× via a ModuleList looped in
    # forward — NOT four inlined copies. The Linear(16,16) constructor therefore
    # appears exactly once (the canonical block), the compression PR #46 promises.
    assert "nn.ModuleList(" in model_py, model_py
    assert "for _ in range(4)" in model_py, model_py
    assert "for _blk in self." in model_py, model_py
    assert model_py.count("in_features=16, out_features=16") == 1, (
        "reused block was inlined instead of factored into one ×4 module:\n" + model_py
    )

    # …and it still round-trips: the exported package runs and reproduces (1,4).
    proc = _run(pkg_dir, "sample_run.py")
    assert proc.returncode == 0, proc.stderr
    assert _printed_shape(proc.stdout) == v["output_shape"]


# ── package is self-contained & wired ─────────────────────────────────────────
def test_export_package_ships_wired_runnable_files(tmp_path):
    """The zip is a self-contained package: the five expected files under
    <name>/, with sample_run.py importing the class model.py actually defines."""
    payload, shape = _mlp()
    pkg_dir, model_py, sample_py = _export_and_extract(payload, shape, str(tmp_path))

    for fname in ("model.py", "__init__.py", "sample_run.py", "README.md", "requirements.txt"):
        assert os.path.isfile(os.path.join(pkg_dir, fname)), f"missing {fname} in package"

    # sample_run.py imports the exact class model.py defines — the wiring that
    # makes `python sample_run.py` resolve without a manual edit.
    assert f"class {MODEL_NAME}(nn.Module)" in model_py, model_py
    assert f"from model import {MODEL_NAME}" in sample_py, sample_py


# ── BUG 1: model.eval() present in model.py __main__, absent in sample_run.py ──
def test_batchnorm_batch1_model_py_main_is_runnable(tmp_path):
    """Companion to the BUG-1 xfail below: proves the model is genuinely valid
    and runnable at batch=1, isolating the defect to sample_run.py. verify_model
    reports (1,10), and model.py's own __main__ block — which DOES call
    model.eval() (code_generation.py _generate_usage_example) — runs and prints
    (1,10). Guards that model.py keeps its eval() so BatchNorm/Dropout use running
    stats at inference."""
    payload, shape = _mlp_batchnorm_batch1()
    v = verify_model(payload, shape)
    assert v["ok"], v.get("errors")
    assert v["output_shape"] == [1, 10]

    pkg_dir, model_py, _sample_py = _export_and_extract(payload, shape, str(tmp_path))
    assert "model.eval()" in model_py  # the __main__ example runs in eval mode

    proc = _run(pkg_dir, "model.py")  # executes model.py's __main__ example
    assert proc.returncode == 0, f"model.py __main__ crashed:\n{proc.stderr}"
    assert _printed_shape(proc.stdout) == [1, 10]


def test_bug_sample_run_batchnorm_batch1_is_runnable(tmp_path):
    """The README quickstart is `python sample_run.py`; for a valid BatchNorm1d
    model at batch=1 it must run and print (1,10). It fails today because the
    generated sample omits model.eval(); this xfail flips to XPASS once fixed."""
    payload, shape = _mlp_batchnorm_batch1()
    pkg_dir, _model_py, _sample_py = _export_and_extract(payload, shape, str(tmp_path))
    proc = _run(pkg_dir, "sample_run.py")
    assert proc.returncode == 0, f"sample_run.py crashed:\n{proc.stderr}"
    assert _printed_shape(proc.stdout) == [1, 10]


# ── BUG 2: verify's hardcoded randint(0,100) false-rejects small-vocab embeds ──
def test_small_vocab_embedding_exported_code_runs(tmp_path):
    """Companion to the BUG-2 xfail below: the EXPORTED code for a small-vocab
    (num_embeddings=50) embedding graph runs correctly, because codegen's
    sample_inputs uses randint(0, num_embeddings). And the identical graph with a
    large vocab (200) already verifies — proving verify_model's rejection of the
    small-vocab case is a false negative in the harness, not a model defect."""
    # Exported small-vocab package runs and produces (2, 8, 4).
    payload50, shape = _embedding(50)
    pkg_dir, _model_py, sample_py = _export_and_extract(payload50, shape, str(tmp_path))
    assert "torch.randint(0, 50" in sample_py, sample_py  # vocab-aware sampling
    proc = _run(pkg_dir, "sample_run.py")
    assert proc.returncode == 0, f"small-vocab exported code crashed:\n{proc.stderr}"
    assert _printed_shape(proc.stdout) == [2, 8, 4]

    # The asymmetry that pins the bug to the hardcoded high=100: same graph, only
    # the vocab differs, yet the large-vocab graph verifies fine.
    v_big = verify_model(*_embedding(200))
    assert v_big["ok"], v_big.get("errors")
    assert v_big["output_shape"] == [2, 8, 4]


def test_bug_verify_small_vocab_embedding_passes():
    """A valid embedding graph must verify regardless of vocab size. With
    num_embeddings=50 verify_model reports ok=False today; flips to XPASS once the
    synthetic input respects num_embeddings."""
    v = verify_model(*_embedding(50))
    assert v["ok"], f"small-vocab embedding false-rejected: {v.get('errors')}"
    assert v["output_shape"] == [2, 8, 4]

#!/usr/bin/env python3
"""Generate 100 real local PR pairs, exercise Git -> HTTP projection, and save Jest evidence.

Only generated models are executed, to check fixture validity. Product review
requests remain source-only. Run with the repository's Torch/test environment.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
FAMILIES = ["mlp", "conv1d", "conv2d", "conv3d", "depthwise", "residual", "autoencoder", "unet",
            "embedding", "attention", "transformer", "lstm", "gru", "rnn", "siamese", "multihead",
            "fusion", "gated", "container", "multifile"]
SIZES = [(8, 1), (16, 2), (24, 6), (32, 16), (48, 48)]
CHANGES = ["keyword_equivalence", "comments_only", "dropout_probability", "activation_replacement",
           "insert_adapter", "remove_dropout", "output_rewire", "attribute_value", "import_alias", "compound_insertion"]


def model(family: str, width: int, depth: int, cls: str):
    """Different families have different computation and valid representative inputs."""
    init, body, extra = [], [], {}
    inputs = {"shape": [2, width], "dtype": "float"}
    for i in range(depth):
        name = f"stage{i}"
        if family in {"mlp", "residual", "siamese", "multihead", "fusion", "gated", "multifile"}:
            op = f"nn.Linear({width}, {width})"
            if family == "multifile":
                op = f"Block({width})"
                extra["blocks.py"] = ("from torch import nn\nclass Block(nn.Module):\n"
                    "    def __init__(self, width):\n        super().__init__()\n"
                    "        self.fc = nn.Linear(width, width)\n        self.norm = nn.LayerNorm(width)\n"
                    "    def forward(self, x):\n        return self.norm(self.fc(x))\n")
            init.append(f"self.{name} = {op}")
            body.append(f"x = self.{name}(x)" + (" + x" if family == "residual" else ""))
        elif family.startswith("conv") or family in {"depthwise", "unet"}:
            dim = int(family[-2]) if family.startswith("conv") else 2
            groups = f", groups={width}" if family == "depthwise" else ""
            init.append(f"self.{name} = nn.Conv{dim}d({width}, {width}, 3, padding=1{groups})")
            body.append(f"x = self.{name}(x)")
            inputs["shape"] = [1, width, *([4] * dim)]
            if family == "unet":
                init.extend([f"self.down{i} = nn.AvgPool2d(2)", f"self.up{i} = nn.Upsample(scale_factor=2, mode='nearest')"])
                body.append(f"x = x + self.up{i}(self.down{i}(x))")
        elif family == "autoencoder":
            init.extend([f"self.encoder{i} = nn.Linear({width}, {width // 2})", f"self.decoder{i} = nn.Linear({width // 2}, {width})"])
            body.append(f"x = self.decoder{i}(self.encoder{i}(x))")
        elif family == "embedding":
            if i == 0:
                init.append(f"self.tokens = nn.Embedding({width * 8}, {width})")
                body.append("x = self.tokens(x)")
            init.append(f"self.{name} = nn.Linear({width}, {width})")
            body.append(f"x = self.{name}(x)")
            inputs = {"shape": [2, 4], "dtype": "long"}
        elif family in {"attention", "transformer"}:
            op = f"nn.MultiheadAttention({width}, 4, batch_first=True)" if family == "attention" else f"nn.TransformerEncoderLayer({width}, 4, dim_feedforward={width * 2}, batch_first=True)"
            init.append(f"self.{name} = {op}")
            body.append(f"x = self.{name}(x, x, x)[0]" if family == "attention" else f"x = self.{name}(x)")
            inputs["shape"] = [2, 4, width]
        elif family in {"lstm", "gru", "rnn"}:
            init.append(f"self.{name} = nn.{family.upper()}({width}, {width}, batch_first=True)")
            body.append(f"x = self.{name}(x)[0]")
            inputs["shape"] = [2, 4, width]
        elif family == "container":
            if i == 0:
                members = ", ".join(f"nn.Linear({width}, {width})" for _ in range(depth))
                init.append(f"self.blocks = nn.ModuleList([{members}])")
                body.extend(["for block in self.blocks:", "    x = block(x)"])
    if family == "siamese":
        body.append("x = x + self.stage0(x)")
    elif family == "multihead":
        init.extend([f"self.left = nn.Linear({width}, {width})", f"self.right = nn.Linear({width}, {width})"])
        body.append("x = self.left(x) + self.right(x)")
    elif family == "fusion":
        init.append(f"self.merge = nn.Linear({width * 2}, {width})")
        body.append("x = self.merge(torch.cat([x, x], dim=-1))")
    elif family == "gated":
        init.append(f"self.gate = nn.Linear({width}, {width})")
        body.append("x = x * torch.sigmoid(self.gate(x))")
    init.extend(["self.activation = nn.ReLU()", "self.dropout = nn.Dropout(0.1)", "self.scale = 1.0"])
    body.extend(["x = self.activation(x)", "return self.dropout(x)"])
    imports = "import torch\nfrom torch import nn\n" + ("from blocks import Block\n" if extra else "")
    source = imports + f"\nclass {cls}(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
    source += "\n".join("        " + line for line in init)
    source += "\n\n    def forward(self, x):\n" + "\n".join("        " + line for line in body) + "\n"
    return {"model.py": source, **extra}, inputs


def mutate(files, kind):
    files = dict(files)
    s = files["model.py"]
    if kind == "keyword_equivalence":
        s = s.replace("nn.Dropout(0.1)", "nn.Dropout(p=0.1)")
    elif kind == "comments_only":
        s = "# PR: documentation only\n\n" + s.replace("self.activation =", "# Activation\n        self.activation =")
    elif kind == "dropout_probability":
        s = s.replace("nn.Dropout(0.1)", "nn.Dropout(0.35)")
    elif kind == "activation_replacement":
        s = s.replace("nn.ReLU()", "nn.GELU()")
    elif kind in {"insert_adapter", "compound_insertion"}:
        s = s.replace("self.dropout =", "self.adapter = nn.Identity()\n        self.dropout =")
        s = s.replace("return self.dropout(x)", "x = self.adapter(x)\n        return self.dropout(x)")
        if kind == "compound_insertion":
            s = s.replace("nn.Dropout(0.1)", "nn.Dropout(0.25)")
    elif kind == "remove_dropout":
        s = s.replace("        self.dropout = nn.Dropout(0.1)\n", "").replace("self.dropout(x)", "x")
    elif kind == "output_rewire":
        # Keep both declarations; only change which layer feeds the output.
        s = s.replace("return self.dropout(x)", "return x")
    elif kind == "attribute_value":
        s = s.replace("self.scale = 1.0", "self.scale = 2.0")
    elif kind == "import_alias":
        s = s.replace("from torch import nn", "from torch import nn as layers").replace("nn.", "layers.")
    files["model.py"] = s
    return files


def git(folder, *args):
    # Our fixture creation must never inherit a real repository's hook context.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.check_output(["git", "-C", str(folder), "-c", "user.name=Graph Review Test", "-c",
        "user.email=graph-review@example.invalid", "-c", "commit.gpgsign=false", *args], env=env, text=True).strip()


def validate_model(files, cls, inputs):
    import torch
    torch.set_num_threads(1)
    helper = sys.modules.get("blocks")
    try:
        if "blocks.py" in files:
            module = types.ModuleType("blocks")
            exec(compile(files["blocks.py"], "blocks.py", "exec"), module.__dict__)
            sys.modules["blocks"] = module
        namespace = {}
        exec(compile(files["model.py"], "model.py", "exec"), namespace)
        net = namespace[cls]().eval()
        x = torch.ones(inputs["shape"], dtype=torch.long if inputs["dtype"] == "long" else torch.float32)
        with torch.inference_mode():
            out = net(x)
        assert torch.isfinite(out).all()
        return {"parameters": sum(p.numel() for p in net.parameters()), "outputShape": list(out.shape)}
    finally:
        if helper is None:
            sys.modules.pop("blocks", None)
        else:
            sys.modules["blocks"] = helper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "canvas/src/test/fixtures/canvas/graphReviewMatrix.json.gz")
    parser.add_argument("--workspace", type=Path, default=ROOT / ".qa-host/graph-review-matrix")
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=args.workspace))
    manifest = []
    for family_index, family in enumerate(FAMILIES):
        for size, (width, depth) in enumerate(SIZES):
            number = family_index * len(SIZES) + size + 1
            cls = f"Review{family.title()}{size + 1}"
            name = f"{number:03d}-{family}-{size + 1}"
            kind = CHANGES[(family_index + 3 * size) % len(CHANGES)]
            before, inputs = model(family, width, depth, cls)
            if kind == "attribute_value":
                before["model.py"] = before["model.py"].replace("return self.dropout(x)", "return self.dropout(x) * self.scale")
            after = mutate(before, kind)
            assert before != after
            repo = run / name
            repo.mkdir()
            git(repo, "init", "-q", "-b", "main")
            for file, source in before.items():
                (repo / file).write_text(source)
            git(repo, "add", ".")
            git(repo, "commit", "-qm", "Base model")
            base = git(repo, "rev-parse", "HEAD")
            git(repo, "checkout", "-qb", "dummy-pr")
            for file, source in after.items():
                (repo / file).write_text(source)
            git(repo, "add", ".")
            git(repo, "commit", "-qm", kind)
            head = git(repo, "rev-parse", "HEAD")
            validity = {"before": validate_model(before, cls, inputs), "after": validate_model(after, cls, inputs)}
            manifest.append({"id": name, "family": family, "size": size + 1, "width": width, "depth": depth,
                "cls": cls, "kind": kind, "repo": str(repo), "base": base, "head": head, "inputs": inputs, "validity": validity})
            print(f"Prepared {name}: {kind}", flush=True)
    manifest_path = run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    source_path = run / "sources.json"
    subprocess.run(["node", "--require", str(ROOT / "extension/out/test/bootstrap.js"),
        str(ROOT / "extension/scripts/graph-review-matrix-sources.cjs"), str(manifest_path), str(source_path)], check=True)
    sources = json.loads(source_path.read_text())
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers import ide
    from auth import CurrentUser, get_current_user
    from services.project_store import ProjectStore
    ide._store = ProjectStore(str(run / "projection.sqlite3"))
    app = FastAPI()
    app.include_router(ide.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(uid="local-dev-user")
    corpus = []
    with TestClient(app) as client:
        for case in sources:
            state = {"id": case["id"], "beforeLabel": case["base"], "afterLabel": case["head"], "stale": False,
                     "changedFiles": case["changedFiles"]}
            for side in ("before", "after"):
                source_set = case[side]
                root = next(e for e in source_set["entries"] if e["file"] == "model.py")
                started = time.perf_counter()
                response = client.post("/api/ide/multifile/project", json={"protocol_version": 1,
                    "root": {"file": "model.py", "cls": case["cls"], "source_digest": root["digest"]},
                    "sources": source_set["entries"], "truncated": source_set["truncated"]})
                if response.status_code != 200:
                    case.setdefault("errors", []).append({"side": side, "status": response.status_code, "detail": response.text})
                    continue
                p = response.json()
                assert p["source_digest"] == root["digest"] and not p["traced"]
                state[side] = {"graphId": p["graph_id"], "revision": p["revision"], "cls": p["cls"], "file": "model.py",
                    "sourceDigest": p["source_digest"], "projectionDigest": p["projection_digest"], "traced": False,
                    "greyed": False, "viewRestored": False, **{k: p[k] for k in ["document", "spanmap", "diagnostics", "dependencies", "truncated"]}}
                case[f"{side}ProjectionMs"] = round((time.perf_counter() - started) * 1000, 2)
            metadata = {k: v for k, v in case.items() if k not in {"before", "after", "repo", "changedFiles"}}
            metadata["sourceLines"] = {side: {e["file"]: len(e["source"].splitlines()) for e in case[side]["entries"]} for side in ("before", "after")}
            corpus.append({**metadata, "state": state})
            print(f"Projected {case['id']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(corpus, separators=(",", ":")).encode(), mtime=0))
    (args.workspace / "latest-run.txt").write_text(str(run))
    print(f"Wrote {len(corpus)} cases to {args.output}; real Git repositories in {run}", flush=True)


if __name__ == "__main__":
    main()

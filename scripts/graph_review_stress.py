#!/usr/bin/env python3
"""Build 1,000 executable model PR branches and project their pinned Git sources."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from graph_review_stress_catalog import cases, validate, FAMILIES, MUTATIONS, SIZES

ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args, input=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.check_output(["git", "-C", str(repo), "-c", "user.name=TensorViz Graph Review QA", "-c", "user.email=graph-review@example.invalid", "-c", "commit.gpgsign=false", *args], input=input, env=env).decode().strip()


def build(repo, state):
    if (state / "manifest.json").exists():
        raise RuntimeError("Corpus already exists; use its pinned manifest instead of rebuilding")
    repo.mkdir(parents=True, exist_ok=False)
    state.mkdir(parents=True, exist_ok=True)
    manifest, changes = [], []
    for item in cases():
        folder = repo / "models" / item["id"]
        folder.mkdir(parents=True)
        validity = {side: validate(item[side], item["cls"], item["inputs"]) for side in ("before", "after")}
        for name, source in item["before"].items(): (folder / name).write_text(source)
        (folder / "pyproject.toml").write_text(f'[project]\nname = "graph-review-{item["id"]}"\nversion = "0.0.0"\n')
        shape = json.dumps(item["inputs"]["shape"])
        (folder / "tensorviz.toml").write_text(f'''version = 1
[profiles.tensorviz_{item["cls"]}]
kind = "model"
model = "model:{item["cls"]}"
cwd = "."
mode = "eval"
device = "cpu"
grad_enabled = false
deterministic = true
seed = 7301
init = {{ args = [], kwargs = {{}} }}
inputs = {{ args = [], kwargs = {{ x = {{ kind = "tensor", shape = {shape}, dtype = "{item["inputs"]["dtype"]}", requires_grad = false, generator = {{kind = "full", value = {1 if item["inputs"]["dtype"] == "int64" else 0.05}}} }} }} }}
''')
        row = {k: v for k, v in item.items() if k not in {"before", "after"}}
        row.update(validity=validity, folder=f"models/{item['id']}", branch=f"case/{item['id']}",
                   sourceDigests={side: {file: hashlib.sha256(src.encode()).hexdigest() for file, src in item[side].items()} for side in ("before", "after")})
        manifest.append(row)
        changes.append({file: source for file, source in item["after"].items() if source != item["before"].get(file)})
        if len(manifest) % 100 == 0: print(f"Validated {len(manifest)}/1000 model pairs", flush=True)
    description = {"models": 1000, "architectures": 100, "sizes": SIZES, "domains": FAMILIES, "mutations": MUTATIONS,
                   "scope": "Small representative PyTorch computation patterns, not trained production checkpoints. Source graph comparisons do not establish runtime equivalence."}
    (repo / "catalog.json").write_text(json.dumps(description, indent=2) + "\n")
    (repo / "README.md").write_text("""# TensorViz graph review stress corpus

1,000 executable PyTorch model cases: 100 architecture patterns at ten sizes and
depths, across eight broad domains. Twenty source mutations each occur 50 times.
Each `case/*` branch changes one model and has a real GitHub pull request.

These are small representative computation patterns, not pretrained checkpoints
or complete implementations of named production systems. Both source revisions
must run a finite CPU forward before publication. Product graph review itself
remains source-only. A pass is an assertion about a stated review contract, not a
claim of arbitrary Python coverage or runtime equivalence.

Open a model folder in VS Code, then use Open as Graph and Review model changes.
Compare its case branch with `main`. Each folder is a separate Python project so
the 1,000-model monorepo does not hide missing dependencies behind source limits.
Root-workspace truncation is tested separately and must remain visible.

The `review-results` branch holds per-PR review evidence tied to GitHub base/head
SHAs, expected mutations, projection diagnostics, and graph comparison results.
Initial failures and unresolved coverage are retained separately from reruns.
No GitHub Actions workflows are installed in this test repository.
""")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Add 1000 validated architecture specimens")
    base = git(repo, "rev-parse", "HEAD")
    stream = bytearray()
    now = int(time.time())
    for item, edits in zip(manifest, changes):
        message = f"{item['id']}: {item['kind']}\n".encode()
        stream.extend(f"commit refs/heads/{item['branch']}\ncommitter TensorViz Graph Review QA <graph-review@example.invalid> {now} +0000\ndata {len(message)}\n".encode() + message + f"from {base}\n".encode())
        for file, source in edits.items():
            data = source.encode()
            stream.extend(f"M 100644 inline {item['folder']}/{file}\ndata {len(data)}\n".encode() + data + b"\n")
        stream.extend(b"\n")
    git(repo, "fast-import", "--quiet", input=bytes(stream))
    for item in manifest:
        item.update(base=base, head=git(repo, "rev-parse", item["branch"]), repo=str(repo / item["folder"]))
    (state / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {len(manifest)} branches at base {base}", flush=True)


def project(repo, state, output):
    sys.path.insert(0, str(ROOT / "backend"))
    manifest = json.loads((state / "manifest.json").read_text())
    for row in manifest: row["repo"] = str(repo / row["folder"])
    inputs = state / "projection-manifest.json"
    inputs.write_text(json.dumps(manifest))
    sources = state / "sources.json"
    subprocess.run(["node", "--require", str(ROOT / "extension/out/test/bootstrap.js"), str(ROOT / "extension/scripts/graph-review-matrix-sources.cjs"), str(inputs), str(sources)], check=True)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers import ide
    from auth import CurrentUser, get_current_user
    from services.project_store import ProjectStore
    ide._store = ProjectStore(str(state / f"projection-{time.time_ns()}.sqlite3"))
    app = FastAPI()
    app.include_router(ide.router)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(uid="local-dev-user")
    corpus = []
    with TestClient(app) as client:
        for case in json.loads(sources.read_text()):
            review = {"id": case["id"], "beforeLabel": case["base"], "afterLabel": case["head"], "stale": False, "changedFiles": case["changedFiles"]}
            for side in ("before", "after"):
                source_set = case[side]
                for entry in source_set["entries"]:
                    assert entry["digest"] == case["sourceDigests"][side][entry["file"]], (case["id"], side, "Git source digest mismatch")
                root = next(e for e in source_set["entries"] if e["file"] == "model.py")
                started = time.perf_counter()
                response = client.post("/api/ide/multifile/project", json={"protocol_version": 1,
                    "root": {"file": "model.py", "cls": case["cls"], "source_digest": root["digest"]},
                    "sources": source_set["entries"], "truncated": source_set["truncated"]})
                case[f"{side}ProjectionMs"] = round((time.perf_counter() - started) * 1000, 2)
                if response.status_code != 200:
                    case.setdefault("errors", []).append({"side": side, "status": response.status_code, "detail": response.text})
                    continue
                p = response.json()
                assert p["source_digest"] == root["digest"] and not p["traced"]
                review[side] = {"graphId": p["graph_id"], "revision": p["revision"], "cls": p["cls"], "file": "model.py", "sourceDigest": p["source_digest"], "projectionDigest": p["projection_digest"], "traced": False, "greyed": False, "viewRestored": False,
                    **{k: p[k] for k in ["document", "spanmap", "diagnostics", "dependencies", "truncated"]}}
            meta = {k: v for k, v in case.items() if k not in {"before", "after", "repo", "changedFiles"}}
            meta["sourceLines"] = {side: {e["file"]: len(e["source"].splitlines()) for e in case[side]["entries"]} for side in ("before", "after")}
            corpus.append({**meta, "state": review})
            if len(corpus) % 50 == 0: print(f"Projected {len(corpus)}/1000", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(json.dumps(corpus, separators=(",", ":")).encode(), mtime=0))
    print(f"Wrote {len(corpus)} review pairs to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "project"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "build": build(args.repo.resolve(), args.state.resolve())
    else: project(args.repo.resolve(), args.state.resolve(), args.output or args.state / "corpus.json.gz")

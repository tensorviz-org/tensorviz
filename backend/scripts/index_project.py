"""Print the project graph the indexer reads out of a folder of Python files.

There is no canvas yet, so this is how you look at what
`docs/project-graph-plan.md` Phase 1 actually produces. It calls the indexer
directly — no backend process, no extension, no network — and nothing it touches
is executed: the whole point of the indexer is that it parses.

    cd <repo root>
    KMP_DUPLICATE_LIB_OK=TRUE ./venv/bin/python backend/scripts/index_project.py examples/tensorviz-research
    KMP_DUPLICATE_LIB_OK=TRUE ./venv/bin/python backend/scripts/index_project.py ~/some/research/repo

`KMP_DUPLICATE_LIB_OK` silences an OpenMP clash when torch loads on macOS, and the
venv interpreter is needed because the repo's system Python is older than the
backend's floor.

What to look for, in the order it usually goes wrong:

* **experiments** — are the real entrypoints here, and is anything listed that is
  plainly a module or a config file?
* **roots** — one per genuinely separate project. Several in a flat repo means
  import resolution disagrees with you.
* **provenance** — `static` was read from the code, `inferred` is a rule guessing.
  A graph carried entirely by `inferred` edges is one to be suspicious of.
* **unreached** — files nothing imports. They should be exactly the ones you expect.

Options:
    --structure   also print contains/imports/calls edges
    --json        dump the whole index instead of the summary
    --mermaid     print one experiment as a Mermaid flowchart, for a README, a bug
                  report, or reading the graph without a browser
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research.document import build_experiment_document  # noqa: E402
from research.export import to_mermaid  # noqa: E402
from research.indexer import index_project  # noqa: E402

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "site-packages",
}

STRUCTURE = {"contains", "imports", "calls"}


def read_sources(root: str) -> dict:
    """Every `.py` under `root`, keyed by its path relative to it."""
    sources = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    sources[os.path.relpath(path, root).replace(os.sep, "/")] = handle.read()
            except OSError:
                # A file that vanished mid-walk is not worth failing the run over.
                continue
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=".", help="folder to index")
    parser.add_argument("--structure", action="store_true", help="show contains/imports/calls")
    parser.add_argument("--json", action="store_true", help="dump the whole index")
    parser.add_argument(
        "--mermaid",
        nargs="?",
        const="",
        metavar="EXPERIMENT",
        help="print one experiment as a Mermaid flowchart (default: the first)",
    )
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    sources = read_sources(root)
    if not sources:
        print(f"no Python files under {root}")
        return 0

    index = index_project(sources)

    if args.json:
        print(json.dumps(index.model_dump(mode="json"), indent=2))
        return 0

    if args.mermaid is not None:
        # The graph as text, which is the only form you can read without a browser
        # — including whoever is fixing the canvas. Paste it into any Mermaid
        # renderer, or a README.
        wanted = args.mermaid
        chosen = next(
            (e for e in index.experiments if wanted in (e.label, e.entrypoint, e.experiment_id)),
            index.experiments[0] if index.experiments else None,
        )
        if chosen is None:
            print("no experiment to draw", file=sys.stderr)
            return 1
        if wanted and wanted not in (chosen.label, chosen.entrypoint, chosen.experiment_id):
            print(f"no experiment named {wanted!r}; drawing {chosen.label}", file=sys.stderr)
        print(to_mermaid(build_experiment_document(chosen.graph, name=chosen.label), title=chosen.label))
        return 0

    print(f"files {len(sources)}   roots {index.roots}   experiments {len(index.experiments)}\n")
    for experiment in sorted(index.experiments, key=lambda e: e.entrypoint):
        flags = experiment.status.state + (", declared" if experiment.status.declared else "")
        print(
            f"{experiment.entrypoint}  [{flags}]  "
            f"root={experiment.root!r}  closure={len(experiment.closure)}"
        )
        by_id = {n.id: n for n in experiment.graph.nodes}
        for node in experiment.graph.nodes:
            if node.kind == "file" and not args.structure:
                continue
            reason = node.params.get("_reason", "")
            print(f"    [{node.kind:13}] {node.label:26} {node.provenance:9} {reason}")
        for edge in experiment.graph.edges:
            if edge.relationship in STRUCTURE and not args.structure:
                continue
            source, target = by_id[edge.from_node_id], by_id[edge.to_node_id]
            print(
                f"    {source.label} --{edge.relationship}--> {target.label} "
                f"({edge.provenance})"
            )
        print()

    if index.unreached:
        print(f"unreached ({len(index.unreached)}):")
        for entry in index.unreached:
            suffix = f"  — {entry.error}" if entry.error else ""
            print(f"    {entry.file}{suffix}")
    if index.truncated:
        print("\nNOTE: the source set was truncated; this project is incomplete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

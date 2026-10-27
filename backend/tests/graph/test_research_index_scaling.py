"""The index must not get quadratic as a workspace grows.

Two defects made it so, and both were invisible in the test corpus: nanoGPT has
15 files, where an O(n^2) insert and a recomputed pure function are free. They
cost 1.3 seconds on a 415-file repository.

These are not timing tests — a clock in a suite is a flake waiting to happen.
They pin the two properties that were violated, by counting calls.
"""

import ast
from typing import Dict

import research.indexer as indexer


def _workspace(files: int, imports_each: int) -> Dict[str, str]:
    sources = {
        "train.py": "import torch\nimport torch.nn as nn\n"
        + "".join(f"from mod{i} import C{i}\n" for i in range(files))
        + "\n".join(f"c{i} = C{i}()" for i in range(files))
    }
    for i in range(files):
        sources[f"mod{i}.py"] = (
            "import torch.nn as nn\n"
            + "".join(f"from mod{j} import C{j}\n" for j in range(min(i, imports_each)))
            + f"\n\nclass C{i}(nn.Module):\n    def forward(self, x):\n        return x\n"
        )
    return sources


def test_source_roots_is_computed_from_the_file_set_not_per_import(monkeypatch):
    """It is a pure function of the file set, so asking it per import asks the same
    question of the same set thousands of times. On 415 files it ran 74,217 times."""
    calls = {"n": 0}
    real = indexer.source_roots

    def counted(files):
        calls["n"] += 1
        return real(files)

    monkeypatch.setattr(indexer, "source_roots", counted)
    sources = _workspace(12, 6)
    indexer.index_project(sources)

    # A handful — one per graph built, plus the index's own — never one per import.
    assert calls["n"] < len(sources), (
        f"source_roots ran {calls['n']} times over {len(sources)} files; it is a pure "
        "function of the file set and belongs outside the import loop"
    )


def test_edge_de_duplication_does_not_scan_every_edge_already_placed():
    """`any(e.id == edge.id for e in edges)` per insert is O(edges^2): 53,666 inserts
    ran that generator 20.5 million times. The check is a set membership test."""
    source = ast.parse(  # the module under test, read rather than imported twice
        (indexer.__file__ and open(indexer.__file__, encoding="utf-8").read()) or ""
    )
    add_edge = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef) and node.name == "add_edge"
    )
    scans = [
        call
        for call in ast.walk(add_edge)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "any"
    ]
    assert not scans, "add_edge must not scan the edge list it is appending to"


def test_a_wider_workspace_costs_proportionally_more_and_not_quadratically():
    """Twice the files must not be four times the work.

    Measured in analyse_file calls rather than seconds: every file is analysed once,
    whatever the import graph does. A file re-analysed per experiment is the shape
    the quadratic defects had.
    """
    small = _workspace(10, 4)
    large = _workspace(20, 4)

    seen = {"n": 0}
    real = indexer.analyse_file

    def counted(*args, **kwargs):
        seen["n"] += 1
        return real(*args, **kwargs)

    indexer.analyse_file = counted  # type: ignore[assignment]
    try:
        seen["n"] = 0
        indexer.index_project(small)
        small_calls = seen["n"]
        seen["n"] = 0
        indexer.index_project(large)
        large_calls = seen["n"]
    finally:
        indexer.analyse_file = real  # type: ignore[assignment]

    assert small_calls <= len(small), "each file is analysed once"
    assert large_calls <= len(large), "each file is analysed once"

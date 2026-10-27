"""The workspace index is bounded, and it indexes the project rather than its libraries.

An editor asks for this index on every open and every save, with no timeout and no
AbortSignal. So an unbounded walk is not merely slow, it is indistinguishable from a
hang — and it was unbounded twice over:

  * **By naming.** The skip list held the two literal names ``venv``/``.venv``. A
    virtualenv called ``env``, ``.env``, ``env310``, a conda prefix or a vendored
    ``site-packages`` was walked in full: 11,075 files, 46 s and 4 GB of resident memory
    on one real tree, and a Models view holding 1,033 torch classes with the user's own
    two models buried in it. The feature was worthless even when it finished.
  * **By size.** With ``root`` at a home directory the walk ran past 180 s and 6 GB and
    was still going after the client had given up.

So the rule is structural — a directory is an environment when the interpreter's own
markers say so — and the walk has a file cap and a deadline it reports hitting.

Run: KMP_DUPLICATE_LIB_OK=TRUE <viz-python> -m pytest tests/test_repo_scan_bounds.py -q
"""

from __future__ import annotations

import os

import pytest


from services.ingest import repo_scan  # noqa: E402


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _project(root: str, *, env_dir: str, marker: str) -> None:
    """A project with one model and a Python environment vendored inside it."""
    _write(
        os.path.join(root, "models", "net.py"),
        "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n",
    )
    env = os.path.join(root, env_dir)
    if marker == "pyvenv.cfg":
        _write(os.path.join(env, "pyvenv.cfg"), "home = /usr/bin\nversion = 3.12.0\n")
        library = os.path.join(env, "lib", "python3.12", "site-packages", "torch")
    elif marker == "site-packages":
        library = os.path.join(env, "site-packages", "torch")
    else:  # conda-meta
        _write(os.path.join(env, "conda-meta", "history"), "\n")
        library = os.path.join(env, "lib", "torch")
    for index in range(40):
        _write(
            os.path.join(library, f"mod{index}.py"),
            f"import torch.nn as nn\n\n\nclass Lib{index}(nn.Module):\n    pass\n",
        )


@pytest.mark.parametrize(
    "env_dir,marker",
    [
        ("env", "pyvenv.cfg"),
        (".env", "pyvenv.cfg"),
        ("env310", "pyvenv.cfg"),
        ("third_party", "site-packages"),
        ("conda", "conda-meta"),
    ],
)
def test_an_environment_is_skipped_however_it_is_named(tmp_path, env_dir, marker) -> None:
    root = str(tmp_path)
    _project(root, env_dir=env_dir, marker=marker)

    scan = repo_scan.scan_repo(root)

    # The user's own model, and only it. Not one row of torch.
    assert sorted(scan.classes) == ["Net"], sorted(scan.classes)[:8]
    assert scan.file_count == 1, f"walked {scan.file_count} files"
    assert scan.truncated is False


def test_a_source_directory_that_merely_sounds_like_one_is_still_scanned(tmp_path) -> None:
    # The rule is the interpreter's markers, not the name. A project directory called
    # `env` that holds source — environment config, an env-var helper — is source, and
    # skipping it would lose the user real classes.
    root = str(tmp_path)
    _write(
        os.path.join(root, "env", "settings.py"),
        "import torch.nn as nn\n\n\nclass EnvNet(nn.Module):\n    pass\n",
    )

    scan = repo_scan.scan_repo(root)

    assert sorted(scan.classes) == ["EnvNet"]


def test_the_file_cap_stops_the_walk_and_says_so(tmp_path) -> None:
    root = str(tmp_path)
    for index in range(30):
        _write(os.path.join(root, "pkg", f"m{index}.py"), "x = 1\n")

    paths, truncated = repo_scan._python_files(root, max_files=10, max_seconds=None)

    assert len(paths) == 10
    # The flag is the point: a partial index the caller is TOLD is partial beats an
    # unbounded one nobody can cancel, but only if the caller can tell.
    assert truncated is True


def test_a_walk_inside_its_bounds_is_not_reported_as_truncated(tmp_path) -> None:
    root = str(tmp_path)
    for index in range(5):
        _write(os.path.join(root, "pkg", f"m{index}.py"), "x = 1\n")

    paths, truncated = repo_scan._python_files(root, max_files=10, max_seconds=None)

    assert len(paths) == 5
    assert truncated is False


def test_the_deadline_stops_a_walk_that_would_not_end(tmp_path, monkeypatch) -> None:
    root = str(tmp_path)
    for index in range(6):
        _write(os.path.join(root, f"pkg{index}", "m.py"), "x = 1\n")

    # Time advances a second per reading, so the deadline trips during the walk rather
    # than depending on how fast this machine happens to be.
    ticks = iter(range(0, 1000))
    monkeypatch.setattr(repo_scan.time, "monotonic", lambda: float(next(ticks)))

    paths, truncated = repo_scan._python_files(root, max_files=None, max_seconds=2.0)

    assert truncated is True
    assert len(paths) < 6, f"the deadline did not cut the walk short: {paths}"


def test_the_shipped_default_bounds_are_real_numbers() -> None:
    # A cap of None or 0 would be the unbounded walk again, wearing a flag.
    assert repo_scan.MAX_SCAN_FILES > 0
    assert repo_scan.MAX_SCAN_SECONDS > 0

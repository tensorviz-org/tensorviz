"""A class that needs an argument is refused — but the repository usually knows the answer.

``# tensorviz: init=Net(hidden=)`` is a skeleton with a BLANK in it, and the blank is the
whole problem: the user is sent to remember a number that ``train.py`` three directories
away states outright. ``repo_scan.resolve_arguments`` has been able to find that number
since Tier B shipped, and the IDE path has never once called it.

So the refusal pre-fills. What it must NOT do is construct: the ladder's best answer is a
guess about intent, and the counter-case ships in this repository —
``examples/tensorviz-sweep/models/resnet_demo.py`` declares ``init=BasicBlock(cin=3,
cout=8)``, a seventh value matching none of its six call sites. These tests pin the three
things that keeps honest: the value is offered and attributed, a repository that
contradicts itself says so instead of picking, and a user who has already written a line
is never overruled by it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pytest


from services.ingest import repo_scan as repo  # noqa: E402
from services.ingest.service import IngestError, ingest_source  # noqa: E402
from services.ingest.spec import ModelRef  # noqa: E402

NET = """
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.fc(x)
"""

FREE = """
import torch.nn as nn


class Free(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.fc(x)
"""


def _repo(tmp_path: Path, source: str, files: Optional[Dict[str, str]] = None) -> Path:
    """A workspace holding ``models/net.py`` plus whatever else the case needs.

    A package, because ``package_root`` walks out of one: the scan root the refusal
    gets is then the workspace, which is where the call sites live.
    """
    package = tmp_path / "models"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    target = package / "net.py"
    target.write_text(source, encoding="utf-8")
    for name, text in (files or {}).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return target


def _refusal(target: Path, cls: str = "Net") -> str:
    with pytest.raises(IngestError) as caught:
        ingest_source(ModelRef(file=str(target), cls=cls))
    return str(caught.value)


def test_the_refusal_offers_the_value_the_repository_already_builds_with(tmp_path: Path):
    """The skeleton arrives filled, and says where the number came from.

    Attribution is not decoration. The user is being asked to ratify a value they did
    not write, and "128, because train.py:3 says so" is a claim they can check in one
    click; "128" alone is indistinguishable from an invention.
    """
    target = _repo(tmp_path, NET, {"train.py": "from models.net import Net\n\nNet(hidden=128)\n"})
    message = _refusal(target)
    assert "# tensorviz: init=Net(hidden=128)" in message
    assert "train.py:3" in message


def test_a_positional_call_site_is_not_invisible(tmp_path: Path):
    """``Net(128)`` answers the question exactly as well as ``Net(hidden=128)``.

    R4's own gate reads ``node.keywords`` only, so an all-positional site contributes
    nothing to it. That is an accident of how the author spelled the call, and it is
    why ``BasicBlock`` — six positional sites — is refused with a blank today.
    """
    target = _repo(tmp_path, NET, {"train.py": "from models.net import Net\n\nNet(128)\n"})
    assert "# tensorviz: init=Net(hidden=128)" in _refusal(target)


def test_call_sites_that_disagree_are_all_named_and_none_is_chosen(tmp_path: Path):
    """Two answers is not one answer with better ranking.

    ``rank_entrypoints`` returns five candidates and lets the caller confirm rather
    than silently auto-picking; a pre-filled constructor argument has strictly less
    room to be wrong in, because the thing that survives the Quick Fix is a line of
    the user's source with no ranking attached to it.
    """
    message = _refusal(
        _repo(
            tmp_path,
            NET,
            {
                "train.py": "from models.net import Net\n\nNet(hidden=128)\n",
                "bench.py": "from models.net import Net\n\nNet(hidden=256)\n",
            },
        )
    )
    assert "# tensorviz: init=Net(hidden=)" in message, "the blank is the honest skeleton here"
    assert "hidden=128" in message and "hidden=256" in message
    assert "train.py:3" in message and "bench.py:3" in message


def test_a_refused_declaration_is_not_overruled_by_the_ladder(tmp_path: Path):
    """The user wrote a line. Their line is the subject, not the repository's habits.

    ``_init_from_comment`` returns ``{}`` both for "no declaration" and for "a
    declaration we refused", so anything chained onto it with ``or`` answers a
    positional declaration with somebody else's value — burying the one diagnostic
    that would have told the user to name their arguments.
    """
    message = _refusal(
        _repo(
            tmp_path,
            NET.replace("class Net", "# tensorviz: init=Net(8)\nclass Net"),
            {"train.py": "from models.net import Net\n\nNet(hidden=128)\n"},
        )
    )
    assert "128" not in message


def test_a_class_that_needs_no_arguments_never_scans_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The cost gate. ``scan_repo`` walks every ``.py`` in the workspace (~0.9 s over
    387 files) and this path runs on every save, so it must not run for the classes
    that construct — which is nearly all of them."""
    def forbidden(root: str):
        raise AssertionError(f"scan_repo({root!r}) ran for a class that needs no arguments")

    monkeypatch.setattr(repo, "scan_repo", forbidden)
    target = _repo(tmp_path, FREE, {"train.py": "from models.net import Free\n\nFree(8)\n"})
    result = ingest_source(ModelRef(file=str(target), cls="Free"))
    assert result.ok


def test_the_proposal_ranks_entry_scripts_first_and_keeps_the_losers(tmp_path: Path):
    """``propose_arguments`` enumerates; it does not collapse.

    The direct test of the ladder's enumerating twin, because the refusal above can
    only observe what survived into one sentence.
    """
    _repo(
        tmp_path,
        NET,
        {
            "tests/test_net.py": "from models.net import Net\n\nNet(hidden=64)\n",
            "train.py": "from models.net import Net\n\nNet(hidden=128)\n",
        },
    )
    proposal = repo.propose_arguments(repo.scan_repo(str(tmp_path)), "Net")
    assert proposal is not None
    assert proposal.kwargs == {"hidden": 128}
    assert "train.py:3" in proposal.provenance
    assert [kwargs for kwargs, _ in proposal.alternatives] == [{"hidden": 64}]


def test_a_class_the_repository_never_instantiates_proposes_nothing(tmp_path: Path):
    """No answer is a real answer, and it must stay distinguishable from a wrong one."""
    _repo(tmp_path, NET)
    assert repo.propose_arguments(repo.scan_repo(str(tmp_path)), "Net") is None

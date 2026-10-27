"""code -> graph -> edit -> code -> graph, and the graph must agree at the end.

Everything a user does on the canvas is a bet that the picture and the file stay
the same model. These are the property tests behind that bet: an edit made on the
graph has to survive being written into source and read back, and it has to change
*only* what the user aimed at. A rename that quietly rewires the network, a
removal that leaves a half-connected layer, or a digest that no longer matches the
buffer are all silent corruption — the user sees a plausible graph and a file that
means something else.

Three shapes are used throughout because the failures differ by shape: a plain
sequential chain, a ModuleList driven by a `for` loop (the control-flow case), and
a branch that fans out and adds back (the multi-input case).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import apply_edits

REPO_ROOT = Path(__file__).resolve().parents[2]

SEQUENTIAL = '''import torch.nn as nn


class Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)
'''

MODULE_LIST = '''import torch.nn as nn


class Loop(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(8, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
'''

RESIDUAL = '''import torch.nn as nn


class Res(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(8, 8)
        self.left = nn.Linear(8, 8)
        self.right = nn.Linear(8, 8)
        self.out = nn.Linear(8, 4)

    def forward(self, x):
        h = self.inp(x)
        a = self.left(h)
        b = self.right(h)
        return self.out(a + b)
'''


def _edges(document):
    return [
        (edge["source_node_id"], edge["target_node_id"])
        for edge in document["semantic"]["edges"]
    ]


def _ids(document):
    return [node["node_id"] for node in document["semantic"]["nodes"]]


def _params(document, node_id):
    for node in document["semantic"]["nodes"]:
        if node["node_id"] == node_id:
            return node["params"]
    raise AssertionError(f"{node_id} is not in the graph: {_ids(document)}")


# --- code -> graph -> set_param -> code -> graph -------------------------------


def test_a_changed_parameter_is_visible_on_the_node_after_the_round_trip(api, edited):
    """The canvas showed 32 units; reopening the file must not show 16 again."""
    file = "round_trip_set_param_seq.py"
    new_source, refused = edited(
        SEQUENTIAL,
        "Seq",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 32}],
        file,
    )
    assert refused == []
    assert "nn.Linear(8, 32)" in new_source

    reparsed = api.parse(new_source, "Seq", file)
    assert reparsed.status_code == 200, reparsed.text
    assert _params(reparsed.json()["document"], "fc1")["out_features"] == 32


def test_editing_one_arm_of_a_branch_leaves_the_other_arm_alone(api, projected, edited):
    """Widening one branch of a residual must not silently widen its twin."""
    file = "round_trip_set_param_res.py"
    before, _spanmap, _digest = projected(RESIDUAL, "Res", file)
    new_source, refused = edited(
        RESIDUAL,
        "Res",
        [{"op": "set_param", "op_id": "o1", "node_id": "left", "param": "out_features", "value": 12}],
        file,
    )
    assert refused == []

    after = api.parse(new_source, "Res", file).json()["document"]
    assert _params(after, "left")["out_features"] == 12
    assert _params(after, "right")["out_features"] == _params(before, "right")["out_features"] == 8
    assert _edges(after) == _edges(before)


def test_editing_a_layer_beside_a_loop_does_not_disturb_the_loop(api, projected, edited):
    """A `for` block is the fragile part of the file; an unrelated edit must miss it."""
    file = "round_trip_set_param_loop.py"
    before, _spanmap, _digest = projected(MODULE_LIST, "Loop", file)
    new_source, refused = edited(
        MODULE_LIST,
        "Loop",
        [{"op": "set_param", "op_id": "o1", "node_id": "head", "param": "out_features", "value": 7}],
        file,
    )
    assert refused == []
    assert "for block in self.blocks:" in new_source

    after = api.parse(new_source, "Loop", file).json()["document"]
    assert _params(after, "head")["out_features"] == 7
    assert _ids(after) == _ids(before)
    assert _edges(after) == _edges(before)


# --- rename must relabel, never rewire ----------------------------------------


@pytest.mark.parametrize(
    "source, cls, old_id, new_id, name",
    [
        (SEQUENTIAL, "Seq", "fc1", "stem", "seq"),
        (RESIDUAL, "Res", "left", "branch_a", "res"),
        (MODULE_LIST, "Loop", "blocks", "stack", "loop"),
    ],
)
def test_a_rename_changes_the_name_and_nothing_about_the_wiring(
    api, projected, edited, source, cls, old_id, new_id, name
):
    """Renaming a layer is cosmetic. If it moves an edge, the model is now a different model."""
    file = f"round_trip_rename_{name}.py"
    before, _spanmap, _digest = projected(source, cls, file)
    new_source, refused = edited(
        source,
        cls,
        [{"op": "rename_layer", "op_id": "o1", "node_id": old_id, "new_node_id": new_id}],
        file,
    )
    assert refused == []

    after = api.parse(new_source, cls, file).json()["document"]
    assert _ids(after) == [new_id if i == old_id else i for i in _ids(before)]
    assert _edges(after) == [
        (new_id if s == old_id else s, new_id if t == old_id else t) for s, t in _edges(before)
    ]


def test_renaming_a_layer_and_renaming_it_back_restores_the_file_exactly(api, projected):
    """Two edits that cancel must leave no residue — otherwise every undo dirties the diff."""
    file = "round_trip_rename_back.py"
    _document, spanmap, digest = projected(SEQUENTIAL, "Seq", file)
    first = api.apply(
        SEQUENTIAL,
        "Seq",
        [{"op": "rename_layer", "op_id": "o1", "node_id": "fc1", "new_node_id": "stem"}],
        spanmap,
        digest,
        file,
    )
    assert first.status_code == 200, first.text
    renamed = apply_edits(SEQUENTIAL, first.json()["edits"])
    assert "self.stem = nn.Linear(8, 16)" in renamed

    second = api.apply(
        renamed,
        "Seq",
        [{"op": "rename_layer", "op_id": "o2", "node_id": "stem", "new_node_id": "fc1"}],
        first.json()["spanmap"],
        first.json()["source_digest"],
        file,
    )
    assert second.status_code == 200, second.text
    assert apply_edits(renamed, second.json()["edits"]) == SEQUENTIAL


# --- remove must not leave a stump --------------------------------------------


def test_removing_a_layer_from_a_chain_closes_the_gap_it_left(api, edited):
    """Delete the activation and fc1 must feed fc2 directly, not feed nothing."""
    file = "round_trip_remove_seq.py"
    new_source, refused = edited(
        SEQUENTIAL, "Seq", [{"op": "remove_layer", "op_id": "o1", "node_id": "act"}], file
    )
    assert refused == []
    assert "self.act" not in new_source

    after = api.parse(new_source, "Seq", file).json()["document"]
    assert _ids(after) == ["input", "fc1", "fc2", "output"]
    assert _edges(after) == [("input", "fc1"), ("fc1", "fc2"), ("fc2", "output")]


def test_removing_one_arm_of_a_branch_keeps_the_join_fed(api, edited):
    """The add still needs two inputs; a removed arm must be bypassed, not orphaned."""
    file = "round_trip_remove_res.py"
    new_source, refused = edited(
        RESIDUAL, "Res", [{"op": "remove_layer", "op_id": "o1", "node_id": "left"}], file
    )
    assert refused == []
    assert "self.left" not in new_source

    after = api.parse(new_source, "Res", file).json()["document"]
    assert _ids(after) == ["input", "inp", "right", "add@Res#0", "out", "output"]
    assert _edges(after) == [
        ("input", "inp"),
        ("inp", "right"),
        ("inp", "add@Res#0"),
        ("right", "add@Res#0"),
        ("add@Res#0", "out"),
        ("out", "output"),
    ]


def test_removing_the_last_layer_returns_the_value_that_reached_it(api, edited):
    """Deleting the head must not leave `return self.fc2(x)` pointing at nothing."""
    file = "round_trip_remove_tail.py"
    new_source, refused = edited(
        SEQUENTIAL, "Seq", [{"op": "remove_layer", "op_id": "o1", "node_id": "fc2"}], file
    )
    assert refused == []
    assert "self.fc2" not in new_source

    after = api.parse(new_source, "Seq", file).json()["document"]
    assert _ids(after) == ["input", "fc1", "act", "output"]
    assert _edges(after) == [("input", "fc1"), ("fc1", "act"), ("act", "output")]


def test_removing_a_looped_modulelist_does_not_leave_a_dangling_reference(edited):
    """A delete on the canvas must never hand back source that cannot run."""
    file = "round_trip_remove_loop.py"
    new_source, refused = edited(
        MODULE_LIST, "Loop", [{"op": "remove_layer", "op_id": "o1", "node_id": "blocks"}], file
    )
    declared = "self.blocks =" in new_source
    used_in_forward = "self.blocks" in new_source.split("def forward")[1]
    assert not (used_in_forward and not declared), (
        f"forward still reads self.blocks with no declaration left; refused={refused}"
    )


# --- idempotence and the digest/spanmap chain ---------------------------------


def test_setting_a_parameter_to_the_value_it_already_has_writes_nothing(api, projected):
    """An edit with no effect must produce no diff — a canvas nudge should not dirty the file."""
    file = "round_trip_noop.py"
    _document, spanmap, digest = projected(SEQUENTIAL, "Seq", file)
    response = api.apply(
        SEQUENTIAL,
        "Seq",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 16}],
        spanmap,
        digest,
        file,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert body["refused"] == []
    assert body["source_digest"] == digest


def test_applying_the_same_edit_twice_changes_nothing_the_second_time(api, projected):
    """Canvas retries and double-clicks happen; the second one must be a no-op, not a second edit."""
    file = "round_trip_idempotent.py"
    op = {"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 32}
    _document, spanmap, digest = projected(SEQUENTIAL, "Seq", file)

    first = api.apply(SEQUENTIAL, "Seq", [op], spanmap, digest, file)
    assert first.status_code == 200, first.text
    once = apply_edits(SEQUENTIAL, first.json()["edits"])

    second = api.apply(
        once, "Seq", [dict(op, op_id="o2")], first.json()["spanmap"], first.json()["source_digest"], file
    )
    assert second.status_code == 200, second.text
    assert second.json()["edits"] == []
    assert apply_edits(once, second.json()["edits"]) == once
    assert second.json()["source_digest"] == first.json()["source_digest"]


@pytest.mark.parametrize(
    "source, cls, op, name",
    [
        (
            SEQUENTIAL,
            "Seq",
            {"op": "set_param", "op_id": "o1", "node_id": "fc2", "param": "out_features", "value": 3},
            "seq",
        ),
        (
            MODULE_LIST,
            "Loop",
            {"op": "rename_layer", "op_id": "o1", "node_id": "stem", "new_node_id": "front"},
            "loop",
        ),
        (
            RESIDUAL,
            "Res",
            {"op": "remove_layer", "op_id": "o1", "node_id": "right"},
            "res",
        ),
    ],
)
def test_the_digest_apply_returns_is_the_digest_of_the_edited_buffer(
    api, projected, source, cls, op, name
):
    """The next edit is gated on this digest; if it lags the buffer, every follow-up edit 409s."""
    file = f"round_trip_digest_{name}.py"
    _document, spanmap, digest = projected(source, cls, file)
    response = api.apply(source, cls, [op], spanmap, digest, file)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["refused"] == []

    new_source = apply_edits(source, body["edits"])
    assert new_source != source

    fresh = api.parse(new_source, cls, file)
    assert fresh.status_code == 200, fresh.text
    assert body["source_digest"] == fresh.json()["source_digest"]
    assert body["spanmap"] == fresh.json()["spanmap"]


def test_two_canvas_edits_in_a_row_work_without_reparsing_in_between(api, projected):
    """This chain is what makes consecutive edits possible: apply's spanmap must feed apply."""
    file = "round_trip_chain.py"
    _document, spanmap, digest = projected(SEQUENTIAL, "Seq", file)

    first = api.apply(
        SEQUENTIAL,
        "Seq",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 32}],
        spanmap,
        digest,
        file,
    )
    assert first.status_code == 200, first.text
    once = apply_edits(SEQUENTIAL, first.json()["edits"])

    second = api.apply(
        once,
        "Seq",
        [{"op": "rename_layer", "op_id": "o2", "node_id": "fc2", "new_node_id": "head"}],
        first.json()["spanmap"],
        first.json()["source_digest"],
        file,
    )
    assert second.status_code == 200, second.text
    twice = apply_edits(once, second.json()["edits"])
    assert "self.fc1 = nn.Linear(8, 32)" in twice
    assert "self.head = nn.Linear(16, 4)" in twice

    after = api.parse(twice, "Seq", file).json()["document"]
    assert _ids(after) == ["input", "fc1", "act", "head", "output"]
    assert _params(after, "fc1")["out_features"] == 32


def test_a_second_edit_carrying_the_first_edits_spans_is_rejected(api, projected):
    """Reusing spans from the pre-edit buffer would write over the wrong lines; 409 is the guard."""
    file = "round_trip_stale.py"
    _document, spanmap, digest = projected(SEQUENTIAL, "Seq", file)
    first = api.apply(
        SEQUENTIAL,
        "Seq",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 32}],
        spanmap,
        digest,
        file,
    )
    once = apply_edits(SEQUENTIAL, first.json()["edits"])

    stale = api.apply(
        once,
        "Seq",
        [{"op": "set_param", "op_id": "o2", "node_id": "fc2", "param": "out_features", "value": 5}],
        spanmap,
        digest,
        file,
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "spanmap_stale"
    assert stale.json()["detail"]["source_digest"] == first.json()["source_digest"]


def test_a_round_trip_edit_never_touches_the_users_disk(api, projected):
    """The edit belongs in the user's dirty buffer and undo stack; the server writing it is data loss."""
    file = "round_trip_never_written.py"
    on_disk = REPO_ROOT / file
    assert not on_disk.exists()

    _document, spanmap, digest = projected(SEQUENTIAL, "Seq", file)
    response = api.apply(
        SEQUENTIAL,
        "Seq",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 64}],
        spanmap,
        digest,
        file,
    )
    assert response.status_code == 200, response.text
    assert "nn.Linear(8, 64)" in apply_edits(SEQUENTIAL, response.json()["edits"])
    assert not on_disk.exists()
    assert not (REPO_ROOT / "backend" / file).exists()

"""Graph → code: what a canvas edit does to the file the user has open.

The whole promise of editing on the canvas is that the file afterwards is one a
reviewer would sign off: the argument the user changed changed, and nothing
else — not a comment, not a blank line, not the author's own spacing. The other
half of the promise is the refusal. A graph is a lossy picture of code, and for
the edits it can draw but source cannot honestly express, the product must say
so by *code* rather than silently write something plausible. These tests assert
the reason codes, never the wording, because the extension branches on the code
and the prose is free to improve.

Nothing here imports the product. Every fact is one an extension could observe.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conftest import apply_edits

BLACKBOX_DIR = Path(__file__).resolve().parent


# A deliberately human file: a docstring, a comment, blank lines and an author's
# own spacing. Everything an edit is not allowed to disturb lives in it.
BASIC = '''import torch.nn as nn


class Net(nn.Module):
    """A tiny net."""

    def __init__(self):
        super().__init__()
        # the first layer
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        return self.fc2(h)
'''


def only_changed_lines(before: str, after: str) -> dict:
    """`{line_number: (old, new)}` for every line that is not byte-identical.

    Written once here because "nothing else changed" is the claim almost every
    test in this file makes, and asserting it by eye in each one would let a
    whitespace regression through.
    """
    old = before.splitlines()
    new = after.splitlines()
    assert len(old) == len(new), f"line count changed: {len(old)} -> {len(new)}"
    return {i + 1: (o, n) for i, (o, n) in enumerate(zip(old, new)) if o != n}


# ── the operations that have a source form ───────────────────────────────────


def test_set_param_rewrites_one_argument_and_leaves_the_file_alone(edited):
    """Widening a layer must not reformat the file it lives in.

    If the product regenerated the module instead of patching the statement, the
    docstring, the comment and the blank lines would all come back subtly
    different and the user's diff would be unreviewable.
    """
    new_source, refused = edited(
        BASIC,
        "Net",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 32}],
        file="g2c_set_param.py",
    )
    assert refused == []
    assert only_changed_lines(BASIC, new_source) == {
        10: ("        self.fc1 = nn.Linear(8, 16)", "        self.fc1 = nn.Linear(8, 32)")
    }


def test_set_param_names_an_argument_the_author_never_wrote(edited):
    """Turning off a bias the author left defaulted has to add the keyword.

    Guessing a positional slot for an argument that is not in the call would put
    the value in the wrong parameter, which still parses and silently builds a
    different model.
    """
    new_source, refused = edited(
        BASIC,
        "Net",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "bias", "value": False}],
        file="g2c_set_param_kw.py",
    )
    assert refused == []
    assert only_changed_lines(BASIC, new_source) == {
        10: (
            "        self.fc1 = nn.Linear(8, 16)",
            "        self.fc1 = nn.Linear(8, 16, bias=False)",
        )
    }


def test_rename_layer_renames_the_declaration_and_its_use_in_forward(edited):
    """A rename that misses `forward` leaves the file at an AttributeError.

    The canvas shows one card, so renaming it must move every `self.<name>` in
    the class, not only the line the card's span points at.
    """
    new_source, refused = edited(
        BASIC,
        "Net",
        [{"op": "rename_layer", "op_id": "o1", "node_id": "fc1", "new_node_id": "stem"}],
        file="g2c_rename.py",
    )
    assert refused == []
    assert only_changed_lines(BASIC, new_source) == {
        10: ("        self.fc1 = nn.Linear(8, 16)", "        self.stem = nn.Linear(8, 16)"),
        15: ("        h = self.fc1(x)", "        h = self.stem(x)"),
    }
    assert "self.fc1" not in new_source


def test_rename_onto_an_existing_layer_is_refused(api, projected):
    """Renaming `fc1` to `fc2` would merge two layers into one attribute.

    Source cannot express that, and writing it would silently delete a layer, so
    the product refuses instead of producing a file with two `self.fc2 = ...`.
    """
    _document, spanmap, digest = projected(BASIC, "Net", "g2c_rename_clash.py")
    response = api.apply(
        BASIC,
        "Net",
        [{"op": "rename_layer", "op_id": "o1", "node_id": "fc1", "new_node_id": "fc2"}],
        spanmap,
        digest,
        "g2c_rename_clash.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["unsupported_operation"]


def test_remove_layer_takes_its_call_in_forward_with_it(edited):
    """Deleting only the declaration leaves `forward` calling a layer that is gone.

    `h = self.act(h)` becomes `h = h` once `act` is deleted, which is not an edit
    a reviewer would accept, so the line goes too.
    """
    new_source, refused = edited(
        BASIC,
        "Net",
        [{"op": "remove_layer", "op_id": "o1", "node_id": "act"}],
        file="g2c_remove.py",
    )
    assert refused == []
    assert "self.act" not in new_source
    remaining = [line for line in BASIC.splitlines() if "self.act" not in line]
    assert new_source.splitlines() == remaining


def test_replace_layer_swaps_the_constructor_and_keeps_the_name(edited):
    """Swapping ReLU for GELU is one changed line, not a rewired `forward`.

    The attribute name is what `forward` calls, so a replacement that renamed it
    would force an edit to a second statement for no reason the user asked for.
    """
    new_source, refused = edited(
        BASIC,
        "Net",
        [
            {
                "op": "replace_layer",
                "op_id": "o1",
                "node_id": "act",
                "new_layer_op": "nn.GELU",
                "params": {},
            }
        ],
        file="g2c_replace.py",
    )
    assert refused == []
    assert only_changed_lines(BASIC, new_source) == {
        11: ("        self.act = nn.ReLU()", "        self.act = nn.GELU()")
    }


def test_replace_layer_can_rename_in_the_same_operation(edited):
    """Replacing a layer and renaming its card is one user gesture, so it is one edit.

    Both halves have to land together: the new constructor and every `self.<old>`
    in `forward`, or the file does not run.
    """
    new_source, refused = edited(
        BASIC,
        "Net",
        [
            {
                "op": "replace_layer",
                "op_id": "o1",
                "node_id": "act",
                "new_layer_op": "nn.GELU",
                "params": {},
                "new_node_id": "gelu",
            }
        ],
        file="g2c_replace_rename.py",
    )
    assert refused == []
    assert only_changed_lines(BASIC, new_source) == {
        11: ("        self.act = nn.ReLU()", "        self.gelu = nn.GELU()"),
        16: ("        h = self.act(h)", "        h = self.gelu(h)"),
    }


def test_a_new_layer_dropped_on_an_edge_is_declared_and_wired(edited):
    """Dragging Dropout between two cards must produce a model that actually runs it.

    A declaration with no call in `forward` is dead weight; a call with no
    declaration is an AttributeError. Both lines, or the gesture was a lie.
    """
    new_source, refused = edited(
        BASIC,
        "Net",
        [
            {
                "op": "add_layer",
                "op_id": "o1",
                "node_id": "drop",
                "layer_op": "nn.Dropout",
                "params": {"p": 0.5},
            },
            {"op": "connect", "op_id": "o2", "source_node_id": "act", "target_node_id": "drop"},
        ],
        file="g2c_add.py",
    )
    assert refused == []
    lines = new_source.splitlines()
    assert lines.index("        self.drop = nn.Dropout(p=0.5)") == lines.index(
        "        self.act = nn.ReLU()"
    ) + 1
    assert lines.index("        h = self.drop(h)") == lines.index("        h = self.act(h)") + 1
    # Everything the user did not touch survives verbatim.
    untouched = [line for line in lines if "drop" not in line]
    assert untouched == BASIC.splitlines()


def test_a_layer_added_but_never_connected_is_refused(api, projected):
    """An unconnected card has no place in `forward`, so there is no line to write.

    Declaring it anyway would leave a layer in `__init__` that never runs — a
    silent no-op the user would read as a successful edit.
    """
    _document, spanmap, digest = projected(BASIC, "Net", "g2c_add_orphan.py")
    response = api.apply(
        BASIC,
        "Net",
        [
            {
                "op": "add_layer",
                "op_id": "o1",
                "node_id": "drop",
                "layer_op": "nn.Dropout",
                "params": {"p": 0.5},
            }
        ],
        spanmap,
        digest,
        "g2c_add_orphan.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["unsupported_operation"]


# ── the refusals, asserted by code and never by wording ──────────────────────


def test_connecting_two_layers_that_already_exist_is_refused(api, projected):
    """Dragging an edge between existing cards is not a one-statement change.

    `forward` decides the order, so honouring it would mean rewriting a function
    body the user never looked at.
    """
    _document, spanmap, digest = projected(BASIC, "Net", "g2c_connect.py")
    response = api.apply(
        BASIC,
        "Net",
        [{"op": "connect", "op_id": "o1", "source_node_id": "fc1", "target_node_id": "fc2"}],
        spanmap,
        digest,
        "g2c_connect.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["unsupported_operation"]


def test_removing_an_edge_on_its_own_is_refused(api, projected):
    """Deleting an arrow is not something source can say; the layers say what runs.

    A refusal is a 200 with a reason — the extension has to show it as "declined",
    not as a failed request, so the status code matters as much as the code.
    """
    _document, spanmap, digest = projected(BASIC, "Net", "g2c_disconnect.py")
    response = api.apply(
        BASIC,
        "Net",
        [
            {
                "op": "remove_connection",
                "op_id": "o1",
                "source_node_id": "fc1",
                "target_node_id": "act",
            }
        ],
        spanmap,
        digest,
        "g2c_disconnect.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["unsupported_operation"]
    assert body["refused"][0]["op_id"] == "o1"


CONTROL_FLOW = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self, big=True):
        super().__init__()
        self.stem = nn.Linear(8, 8)
        if big:
            self.head = nn.Linear(8, 16)
        else:
            self.head = nn.Linear(8, 4)

    def forward(self, x):
        h = self.stem(x)
        return self.head(h)
'''


def test_a_layer_inside_a_branch_is_refused(api, projected):
    """The canvas draws one `head`; the file has two, and only one is reachable.

    Patching the span would change one branch of a card the user sees as whole,
    so the two configurations would silently disagree.
    """
    _document, spanmap, digest = projected(CONTROL_FLOW, "Net", "g2c_control_flow.py")
    assert "head" in spanmap, "the branch layer must be addressable, or this proves nothing"
    response = api.apply(
        CONTROL_FLOW,
        "Net",
        [
            {
                "op": "set_param",
                "op_id": "o1",
                "node_id": "head",
                "param": "out_features",
                "value": 32,
            }
        ],
        spanmap,
        digest,
        "g2c_control_flow.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["control_flow_region"]


def test_a_layer_inside_a_branch_is_still_refused_for_removal(api, projected):
    """Deleting it is the same problem: one branch would lose a layer the other keeps."""
    _document, spanmap, digest = projected(CONTROL_FLOW, "Net", "g2c_control_flow_rm.py")
    response = api.apply(
        CONTROL_FLOW,
        "Net",
        [{"op": "remove_layer", "op_id": "o1", "node_id": "head"}],
        spanmap,
        digest,
        "g2c_control_flow_rm.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["control_flow_region"]


DYNAMIC = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        setattr(self, "hidden", nn.Linear(8, 8))
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        h = self.hidden(x)
        return self.head(h)
'''


def test_a_layer_built_at_runtime_has_nothing_to_patch(api, projected):
    """`setattr` registers a real submodule named `hidden`, but no line declares it.

    A traced graph knows the module by that name, so the canvas can ask to edit
    it. There is no `self.hidden = ...` to patch, and inventing one would add a
    second, shadowing declaration.
    """
    _document, spanmap, digest = projected(DYNAMIC, "Net", "g2c_dynamic.py")
    assert "hidden" not in spanmap, "a declared span here would make this test vacuous"
    response = api.apply(
        DYNAMIC,
        "Net",
        [
            {
                "op": "set_param",
                "op_id": "o1",
                "node_id": "hidden",
                "param": "out_features",
                "value": 4,
            }
        ],
        spanmap,
        digest,
        "g2c_dynamic.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["dynamic_construction"]


NON_LITERAL = '''import torch.nn as nn

WIDTH = 16


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, WIDTH)

    def forward(self, x):
        return self.fc(x)
'''


def test_an_argument_written_as_an_expression_is_not_overwritten(api, projected):
    """`nn.Linear(8, WIDTH)` says the author wanted one knob for the whole file.

    Writing `32` over `WIDTH` would delete that indirection without saying so, so
    the product refuses and hands back the range to reveal instead.
    """
    _document, spanmap, digest = projected(NON_LITERAL, "Net", "g2c_non_literal.py")
    response = api.apply(
        NON_LITERAL,
        "Net",
        [
            {
                "op": "set_param",
                "op_id": "o1",
                "node_id": "fc",
                "param": "out_features",
                "value": 32,
            }
        ],
        spanmap,
        digest,
        "g2c_non_literal.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    refusal = body["refused"][0]
    assert refusal["reason"] == "non_literal_value"
    # The range must point at `WIDTH` itself, because the IDE reveals it.
    assert refusal["span"]["start"] == [9, 31]
    assert refusal["span"]["end"] == [9, 36]
    assert NON_LITERAL.splitlines()[8][31:36] == "WIDTH"


def test_an_argument_the_author_left_literal_is_still_editable(edited):
    """The refusal above is about the expression, not about the whole call.

    A file with one symbolic argument must not become read-only; the literal
    argument next to it still edits.
    """
    new_source, refused = edited(
        NON_LITERAL,
        "Net",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc", "param": "in_features", "value": 4}],
        file="g2c_non_literal_ok.py",
    )
    assert refused == []
    assert only_changed_lines(NON_LITERAL, new_source) == {
        9: ("        self.fc = nn.Linear(8, WIDTH)", "        self.fc = nn.Linear(4, WIDTH)")
    }


def test_an_operation_the_server_does_not_know_is_refused_not_a_crash(api, projected):
    """A newer extension sending an op this backend predates must degrade, not 500.

    The canvas keeps its old state on a refusal; a 500 would look like the file
    was edited when it was not.
    """
    _document, spanmap, digest = projected(BASIC, "Net", "g2c_unknown_op.py")
    response = api.apply(
        BASIC,
        "Net",
        [{"op": "teleport_layer", "op_id": "o1", "node_id": "fc1"}],
        spanmap,
        digest,
        "g2c_unknown_op.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["unsupported_operation"]


# ── the guarantees around an apply, not inside it ────────────────────────────


def test_a_stale_digest_is_rejected_whole(api, projected):
    """Spans from an older buffer point at the wrong lines; a partial apply corrupts.

    The user typed while the canvas was open. Every span in the request is now
    off, so there is nothing to salvage and the answer is 409, not a best effort.
    """
    _document, spanmap, _digest = projected(BASIC, "Net", "g2c_stale.py")
    stale_digest = hashlib.sha256(b"a different buffer entirely").hexdigest()
    response = api.apply(
        BASIC,
        "Net",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 32}],
        spanmap,
        stale_digest,
        "g2c_stale.py",
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "spanmap_stale"


def test_apply_writes_nothing_to_disk(api, projected):
    """The edit belongs in the user's undo stack, not written behind their back.

    The extension turns the returned ranges into a WorkspaceEdit. If the backend
    also wrote the file, the user would have a change they cannot undo and a
    dirty buffer fighting it.
    """
    before = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in sorted(BLACKBOX_DIR.rglob("*"))
        if path.is_file()
    }
    named = BLACKBOX_DIR / "g2c_disk_probe.py"
    assert not named.exists()

    _document, spanmap, digest = projected(BASIC, "Net", str(named))
    response = api.apply(
        BASIC,
        "Net",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 32}],
        spanmap,
        digest,
        str(named),
    )
    assert response.status_code == 200, response.text
    assert response.json()["edits"] == [
        {
            "range": {"start": [10, 0], "end": [11, 0]},
            "newText": "        self.fc1 = nn.Linear(8, 32)\n",
        }
    ]

    assert not named.exists(), "apply created the file it was told the source came from"
    after = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in sorted(BLACKBOX_DIR.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_a_partly_applicable_batch_applies_the_part_it_can(api, projected):
    """One impossible op must not throw away the four the user also made.

    A canvas gesture is often a batch; failing it whole would lose real work over
    an edge the graph could draw and source cannot express.
    """
    _document, spanmap, digest = projected(BASIC, "Net", "g2c_partial.py")
    response = api.apply(
        BASIC,
        "Net",
        [
            {
                "op": "set_param",
                "op_id": "ok",
                "node_id": "fc1",
                "param": "out_features",
                "value": 32,
            },
            {
                "op": "remove_connection",
                "op_id": "no",
                "source_node_id": "fc1",
                "target_node_id": "act",
            },
        ],
        spanmap,
        digest,
        "g2c_partial.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["op_id"] for item in body["refused"]] == ["no"]
    assert [item["reason"] for item in body["refused"]] == ["unsupported_operation"]
    assert only_changed_lines(BASIC, apply_edits(BASIC, body["edits"])) == {
        10: ("        self.fc1 = nn.Linear(8, 16)", "        self.fc1 = nn.Linear(8, 32)")
    }


def test_the_spanmap_an_apply_returns_anchors_the_next_edit(api, projected):
    """A user makes two edits in a row; the second must not need a re-projection.

    Apply returns the digest and spans of the buffer it just produced. If those
    were the old ones, every second edit in a session would 409.
    """
    _document, spanmap, digest = projected(BASIC, "Net", "g2c_chain.py")
    first = api.apply(
        BASIC,
        "Net",
        [{"op": "set_param", "op_id": "o1", "node_id": "fc1", "param": "out_features", "value": 32}],
        spanmap,
        digest,
        "g2c_chain.py",
    )
    assert first.status_code == 200, first.text
    once = apply_edits(BASIC, first.json()["edits"])

    second = api.apply(
        once,
        "Net",
        [{"op": "set_param", "op_id": "o2", "node_id": "fc2", "param": "in_features", "value": 32}],
        first.json()["spanmap"],
        first.json()["source_digest"],
        "g2c_chain.py",
    )
    assert second.status_code == 200, second.text
    assert second.json()["refused"] == []
    twice = apply_edits(once, second.json()["edits"])
    assert only_changed_lines(BASIC, twice) == {
        10: ("        self.fc1 = nn.Linear(8, 16)", "        self.fc1 = nn.Linear(8, 32)"),
        12: ("        self.fc2 = nn.Linear(16, 2)", "        self.fc2 = nn.Linear(32, 2)"),
    }


# ── the graph is data; the file is code ──────────────────────────────────────


def test_a_value_from_the_graph_is_written_as_data_never_as_code(edited):
    """A node's parameter value arrives over HTTP and lands in the user's file.

    If it were interpolated as source text, anything that reaches the graph —
    a shared project, a trace worker, a pasted card — would run in the file the
    user opens next. It has to come out as a quoted literal.
    """
    hostile = "16) or __import__('os').system('id'"
    new_source, refused = edited(
        BASIC,
        "Net",
        [
            {
                "op": "set_param",
                "op_id": "o1",
                "node_id": "fc1",
                "param": "out_features",
                "value": hostile,
            }
        ],
        file="g2c_injection.py",
    )
    assert refused == []
    changed = only_changed_lines(BASIC, new_source)
    assert list(changed) == [10]
    assert changed[10][1] == (
        '        self.fc1 = nn.Linear(8, "16) or __import__(\'os\').system(\'id\'")'
    )
    assert "__import__('os').system('id')" not in new_source


# ── a Sequential's children are addressable, but not deletable ───────────────


SEQUENTIAL = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
'''


def test_a_sequential_child_can_still_be_retuned(api, projected, edited):
    """The canvas draws the children, so editing one has to reach inside the call.

    A user who sees `Linear(8, 16)` as a card expects widening it to work even
    though the layer has no `self.<name>` of its own.
    """
    _document, spanmap, _digest = projected(SEQUENTIAL, "Net", "g2c_seq_param.py")
    assert "linear@Net#0" in spanmap
    new_source, refused = edited(
        SEQUENTIAL,
        "Net",
        [
            {
                "op": "set_param",
                "op_id": "o1",
                "node_id": "linear@Net#0",
                "param": "out_features",
                "value": 32,
            }
        ],
        file="g2c_seq_param.py",
    )
    assert refused == []
    assert only_changed_lines(SEQUENTIAL, new_source) == {
        8: ("            nn.Linear(8, 16),", "            nn.Linear(8, 32),")
    }


def test_deleting_a_sequential_child_is_refused(api, projected):
    """It has no statement of its own; deleting one would delete the whole Sequential.

    Losing two layers when the user asked to lose one is worse than declining.
    """
    _document, spanmap, digest = projected(SEQUENTIAL, "Net", "g2c_seq_remove.py")
    response = api.apply(
        SEQUENTIAL,
        "Net",
        [{"op": "remove_layer", "op_id": "o1", "node_id": "linear@Net#0"}],
        spanmap,
        digest,
        "g2c_seq_remove.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["unsupported_operation"]


LOOP_IN_FORWARD = '''import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Linear(8, 8)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        for _ in range(3):
            x = self.block(x)
        return self.head(x)
'''


def test_a_layer_called_in_a_loop_cannot_be_deleted(api, projected):
    """Dropping `x = self.block(x)` out of a loop changes what the loop does.

    The declaration is fine to delete; the call is not, and the refusal has to
    follow the call site rather than the declaration.
    """
    _document, spanmap, digest = projected(LOOP_IN_FORWARD, "Net", "g2c_loop_remove.py")
    response = api.apply(
        LOOP_IN_FORWARD,
        "Net",
        [{"op": "remove_layer", "op_id": "o1", "node_id": "block"}],
        spanmap,
        digest,
        "g2c_loop_remove.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["control_flow_region"]


def test_a_layer_called_in_a_loop_can_still_be_retuned(edited):
    """Being called three times does not make the constructor unsafe to edit.

    The control-flow guard exists for edits that change one branch of a card, and
    a constructor change is the same for every iteration — refusing it here would
    make a whole class of ordinary models read-only.
    """
    new_source, refused = edited(
        LOOP_IN_FORWARD,
        "Net",
        [{"op": "set_param", "op_id": "o1", "node_id": "block", "param": "out_features", "value": 16}],
        file="g2c_loop_param.py",
    )
    assert refused == []
    assert only_changed_lines(LOOP_IN_FORWARD, new_source) == {
        7: ("        self.block = nn.Linear(8, 8)", "        self.block = nn.Linear(8, 16)")
    }


# ── findings: the product accepts an edit it cannot honestly make ────────────


DYNAMIC_RENAME_TARGET = DYNAMIC


def test_rename_refuses_the_same_layer_set_param_refuses(api, projected):
    """A layer with no declaration is not renameable either, and for the same reason.

    `set_param` on `hidden` refuses `dynamic_construction`, and `rename_layer` on
    the very same node must too. A rename that applied here would rewrite
    `self.hidden(x)` to `self.mid(x)` while `setattr(self, "hidden", ...)` stayed
    put: the file the user is handed would raise AttributeError on the first
    forward pass, and the canvas would have been told the edit succeeded.
    """
    _document, spanmap, digest = projected(DYNAMIC_RENAME_TARGET, "Net", "g2c_dyn_rename.py")
    tuning = api.apply(
        DYNAMIC_RENAME_TARGET,
        "Net",
        [{"op": "set_param", "op_id": "o1", "node_id": "hidden", "param": "out_features", "value": 4}],
        spanmap,
        digest,
        "g2c_dyn_rename.py",
    )
    assert [item["reason"] for item in tuning.json()["refused"]] == ["dynamic_construction"]

    renaming = api.apply(
        DYNAMIC_RENAME_TARGET,
        "Net",
        [{"op": "rename_layer", "op_id": "o1", "node_id": "hidden", "new_node_id": "mid"}],
        spanmap,
        digest,
        "g2c_dyn_rename2.py",
    )
    assert renaming.status_code == 200, renaming.text
    body = renaming.json()
    assert [item["reason"] for item in body["refused"]] == ["dynamic_construction"]
    assert body["edits"] == []


@pytest.mark.parametrize("keyword", ["class", "import", "None", "lambda", "return"])
def test_renaming_to_a_python_keyword_is_refused_not_a_crash(api, projected, keyword):
    """A card renamed to `class` is a typo, not a server error.

    A 500 would give the user no explanation and no way to tell whether their file
    was touched. Every other impossible edit in this API comes back as a refusal
    the extension can show next to the card, and this one must too.
    """
    _document, spanmap, digest = projected(BASIC, "Net", f"g2c_keyword_{keyword}.py")
    response = api.apply(
        BASIC,
        "Net",
        [{"op": "rename_layer", "op_id": "o1", "node_id": "fc1", "new_node_id": keyword}],
        spanmap,
        digest,
        f"g2c_keyword_{keyword}.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["unsupported_operation"]


def test_replacing_a_layer_with_a_class_that_does_not_exist_is_refused(api, projected):
    """Writing `nn.NotAThing()` gives the user a file that cannot even be imported.

    The API already declines `nn.Linear()` for missing arguments on the stated
    grounds that source which parses but cannot run is worse than a refusal. A
    class that does not exist fails the same way, one line later.
    """
    _document, spanmap, digest = projected(BASIC, "Net", "g2c_bogus_class.py")
    response = api.apply(
        BASIC,
        "Net",
        [
            {
                "op": "replace_layer",
                "op_id": "o1",
                "node_id": "act",
                "new_layer_op": "nn.NotAThing",
                "params": {},
            }
        ],
        spanmap,
        digest,
        "g2c_bogus_class.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["unsupported_operation"]


@pytest.mark.xfail(
    strict=True,
    reason="FINDING: set_param writes any JSON scalar into the slot, so a string "
    "lands where an int belongs and `nn.Linear(8, \"wide\")` is reported as a "
    "successful edit.",
)
def test_a_value_of_the_wrong_type_is_refused(api, projected):
    """`out_features=\"wide\"` builds a model that raises the moment it is constructed.

    The product already refuses to write a constructor it knows cannot be built.
    A value the parameter cannot take is the same failure, and the user finds out
    at the next run rather than at the edit.
    """
    _document, spanmap, digest = projected(BASIC, "Net", "g2c_wrong_type.py")
    response = api.apply(
        BASIC,
        "Net",
        [
            {
                "op": "set_param",
                "op_id": "o1",
                "node_id": "fc1",
                "param": "out_features",
                "value": "wide",
            }
        ],
        spanmap,
        digest,
        "g2c_wrong_type.py",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["edits"] == []
    assert [item["reason"] for item in body["refused"]] == ["unsupported_operation"]

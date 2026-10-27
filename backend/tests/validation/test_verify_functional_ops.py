"""The dry forward pass must run a torch.* op, not try to construct it.

Stage 2 of verify_model builds an nn.Module per layer node and runs a synthetic
tensor through the graph. Its only way to run a node was to build a class, so every
functional tensor op — 44 of them in the catalogue, everything named `torch.*` plus
the bare `interpolate` — failed on sight:

    build error: Unknown layer type 'torch.narrow':
                 Layer type 'torch.narrow' not found in torch.nn

Reported on the card while the shape rows above it were correct, because shape
inference and this forward are separate engines and only one of them knew about the
op. Merge ops (add/cat/stack/sum) were exempted by name years ago; nothing exempted
the rest.

Two halves are pinned here, and the second matters as much as the first: it is not
enough for verify to stop erroring. It has to APPLY the op, or a Slice would pass
the unsliced tensor on and the pass would go green on a graph that cannot run.
"""

from __future__ import annotations



import torch

from services.verification_service import (
    _functional_output_shape,
    _is_functional_op_name,
    verify_model,
)


def _node(node_id: str, node_type: str, op_id: str, op_name: str, params: dict, x: int):
    """A node the Graph model will actually accept — `position`, `layerType` and
    `parameters` are all required, and shape inference reports a pydantic wall
    rather than reaching the forward if any is missing."""
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": float(x), "y": 0.0},
        "data": {
            "layerType": {"id": op_id, "name": op_name, "parameters": []},
            "parameters": params,
        },
    }


def _slice_payload(dim: int = -1, start: int = 0, length: int = 16):
    """Input(1, 32) -> Slice -> Linear(length -> 8) -> Output.

    The Linear is what makes this a real test: its in_features must match what the
    Slice produced, so a forward that quietly skipped the Slice would fail here.
    """
    return {
        "nodes": [
            _node("input", "inputNode", "input", "Input", {"shape": "1,32"}, 0),
            _node("slice", "layerNode", "narrow", "torch.narrow",
                  {"dim": dim, "start": start, "length": length}, 200),
            _node("lin", "layerNode", "linear", "nn.Linear",
                  {"in_features": length, "out_features": 8}, 400),
            _node("output", "outputNode", "output", "Output", {}, 600),
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "slice"},
            {"id": "e2", "source": "slice", "target": "lin"},
            {"id": "e3", "source": "lin", "target": "output"},
        ],
    }


class TestFunctionalOpsVerify:
    def test_slice_no_longer_reports_a_build_error(self):
        result = verify_model(graph_payload=_slice_payload(), input_shape=[1, 32], run_forward=True)
        messages = " | ".join(str(e.get("message", e)) for e in result.get("errors", []))
        assert "not found in torch.nn" not in messages, messages
        assert "build error" not in messages, messages
        assert result.get("ok") is True, f"errors: {result.get('errors')}"

    def test_the_slice_is_actually_applied(self):
        """The Linear declares the POST-slice width. If the forward passed the
        unsliced tensor through, torch would raise a mat1/mat2 mismatch here — so a
        green result is only reachable by really narrowing 32 down to 16."""
        result = verify_model(graph_payload=_slice_payload(length=16), input_shape=[1, 32], run_forward=True)
        assert result.get("ok") is True, f"errors: {result.get('errors')}"

        # And the converse: a Linear wired for the UNSLICED width must fail, which is
        # what proves the tensor really changed size rather than the check being lax.
        wrong = _slice_payload(length=16)
        wrong["nodes"][2]["data"]["parameters"]["in_features"] = 32
        bad = verify_model(graph_payload=wrong, input_shape=[1, 32], run_forward=True)
        assert bad.get("ok") is not True
        assert bad.get("errors"), "a Linear expecting 32 after a slice to 16 must fail"

    def test_an_impossible_slice_is_reported_not_guessed(self):
        # start + length runs off the end of a 32-wide axis. The op registry declines,
        # and verify must say so rather than inventing an output shape.
        result = verify_model(
            graph_payload=_slice_payload(start=24, length=16), input_shape=[1, 32], run_forward=True
        )
        assert result.get("ok") is not True
        messages = " | ".join(str(e.get("message", e)) for e in result.get("errors", []))
        assert "shape error" in messages or "narrow" in messages.lower(), messages

    def test_reshape_and_permute_verify_too(self):
        """The fix is categorical, not one entry on a list — Slice was simply the op
        that got reported. These two are the other tensor ops a real graph reaches
        for most often."""
        payload = {
            "nodes": [
                _node("input", "inputNode", "input", "Input", {"shape": "1,3,8,8"}, 0),
                _node("perm", "layerNode", "permute", "torch.permute", {"dims": [0, 2, 3, 1]}, 200),
                _node("output", "outputNode", "output", "Output", {}, 400),
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "perm"},
                {"id": "e2", "source": "perm", "target": "output"},
            ],
        }
        result = verify_model(graph_payload=payload, input_shape=[1, 3, 8, 8], run_forward=True)
        messages = " | ".join(str(e.get("message", e)) for e in result.get("errors", []))
        assert "not found in torch.nn" not in messages, messages
        assert result.get("ok") is True, f"errors: {result.get('errors')}"
        # And the permutation really happened: (1,3,8,8) -> (1,8,8,3).
        assert result.get("shapes", {}).get("perm") == [1, 8, 8, 3], result.get("shapes")


class TestFunctionalClassifier:
    def test_recognises_the_torch_namespace_but_not_nn(self):
        assert _is_functional_op_name("torch.narrow")
        assert _is_functional_op_name("torch.reshape")
        # The bare canonical: graph_input_safety's _LAYER_OPERATION guard allows at
        # most one prefix level, so "torch.nn.functional.interpolate" is rejected on
        # purpose and the bare name has to be recognised explicitly.
        assert _is_functional_op_name("interpolate")
        # An nn class is NOT functional, even though its name starts with "torch.".
        assert not _is_functional_op_name("torch.nn.Linear")
        assert not _is_functional_op_name("nn.Linear")
        assert not _is_functional_op_name("Linear")

    def test_output_shape_comes_from_the_op_registry(self):
        # Same shape_fn shape inference and validation use, so the dry forward cannot
        # report a shape the card disagrees with.
        shape = _functional_output_shape(
            {"id": "narrow", "name": "torch.narrow"},
            {"dim": -1, "start": 0, "length": 12},
            [torch.zeros(1, 32)],
        )
        assert shape == [1, 12]
        assert shape == list(torch.narrow(torch.zeros(1, 32), -1, 0, 12).shape)

    def test_declines_rather_than_guessing(self):
        assert _functional_output_shape(
            {"id": "narrow", "name": "torch.narrow"},
            {"dim": -1, "start": 30, "length": 8},
            [torch.zeros(1, 32)],
        ) is None


class TestUnsetParametersAreNamed:
    """A functional op that cannot run because an argument was never set must say
    WHICH argument. The shape engine's decline is indistinguishable from an
    impossible-but-complete slice, and "cannot be applied with these parameters" is
    not something a user can act on.

    The tables come from code_generation, so a node that fails verify for this
    reason fails Export for the same reason and with the same words — the two used
    to disagree: Export refused a palette permute while verify called it fine.
    """

    def test_permute_without_dims_names_dims(self):
        payload = {
            "nodes": [
                _node("input", "inputNode", "input", "Input", {"shape": "1,3,8,8"}, 0),
                # What the palette actually produces: the catalogue exposes this
                # parameter as `dim`, while torch.permute and the shape engine both
                # want `dims`. That mismatch is a separate bug; this pins the
                # message the user gets meanwhile.
                _node("perm", "layerNode", "permute", "torch.permute", {"dim": ""}, 200),
                _node("output", "outputNode", "output", "Output", {}, 400),
            ],
            "edges": [
                {"id": "e1", "source": "input", "target": "perm"},
                {"id": "e2", "source": "perm", "target": "output"},
            ],
        }
        result = verify_model(graph_payload=payload, input_shape=[1, 3, 8, 8], run_forward=True)
        assert result.get("ok") is not True
        messages = " | ".join(str(e.get("message", e)) for e in result.get("errors", []))
        assert "needs dims" in messages, messages
        assert "Properties" in messages, messages

    def test_a_complete_but_impossible_slice_still_gets_the_shape_message(self):
        # Nothing is unset here — start + length simply runs off the axis — so the
        # generic message is the accurate one and must not be replaced.
        result = verify_model(
            graph_payload=_slice_payload(start=24, length=16), input_shape=[1, 32], run_forward=True
        )
        messages = " | ".join(str(e.get("message", e)) for e in result.get("errors", []))
        assert "shape error" in messages, messages
        assert "needs" not in messages, messages

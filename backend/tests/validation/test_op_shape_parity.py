"""Every shape function is checked against what torch actually does.

This is the test that was missing. Five registered ops used to sit on
``_shape_passthrough`` — reporting their input shape verbatim — while genuinely
changing it:

    torch.sum(dim=1)  on [1,64,32,32]  reported [1,64,32,32]  really [1,32,32]
    torch.reshape                      reported [1,64,32,32]  really [1,64,1024]
    torch.split(16, dim=1)             reported [1,64,32,32]  really [1,16,32,32]
    torch.chunk(4, dim=1)              reported [1,64,32,32]  really [1,16,32,32]
    torch.stack(dim=0)                 reported [1,64,32,32]  really [2,1,64,32,32]

Nothing flagged those as unreliable — ``used_passthrough`` is only set for
members of the passthrough *set*, not for a spec whose ``shape_fn`` happens to be
``_shape_passthrough`` — so a wrong shape propagated through every downstream node
as though it had been verified.

A test that asserted a shape function against itself would have passed the whole
time. So each case below runs the real torch op and compares.
"""
from __future__ import annotations

import pytest
import torch

from core.op_registry import spec_for
from core.shape_inference import _apply_analytic_merge, build_layer

# One concrete input used throughout, so the numbers in the docstring above and
# in the failure output line up with each other.
INP = [1, 64, 32, 32]


def _t() -> torch.Tensor:
    return torch.zeros(*INP)


def _reported(op: str, params: dict, all_inputs=None):
    spec = spec_for(op)
    assert spec is not None, f"{op} is not registered"
    return spec.shape_fn(params, list(INP), all_inputs or [list(INP)])


# ── ops whose shape function must equal real torch ─────────────────────────────
# (label, registry name, node params, callable producing the real output)
PARITY = [
    ("reshape to an explicit shape",
     "torch.reshape", {"shape": "1,64,1024"},
     lambda t: torch.reshape(t, (1, 64, 1024))),
    ("reshape inferring one -1",
     "torch.reshape", {"shape": "1,-1"},
     lambda t: torch.reshape(t, (1, -1))),
    ("mean over a dim",
     "torch.mean", {"dim": 1},
     lambda t: torch.mean(t, dim=1)),
    ("mean with keepdim",
     "torch.mean", {"dim": 1, "keepdim": True},
     lambda t: torch.mean(t, dim=1, keepdim=True)),
    ("squeeze a size-1 dim",
     "torch.squeeze", {"dim": 0},
     lambda t: torch.squeeze(t, dim=0)),
    ("unsqueeze",
     "torch.unsqueeze", {"dim": 2},
     lambda t: torch.unsqueeze(t, dim=2)),
    # NOTE the param name: this app stores transpose's axes as a 2-list under
    # `dims`, not as torch's positional dim0/dim1. See
    # test_transpose_param_name_does_not_match_torchs_signature below.
    ("transpose two dims",
     "torch.transpose", {"dims": [1, 3]},
     lambda t: torch.transpose(t, 1, 3)),
    ("permute",
     "torch.permute", {"dims": [0, 2, 3, 1]},
     lambda t: torch.permute(t, (0, 2, 3, 1))),
    # …and as the Properties panel really stores them. That field is free text, so
    # a typed axis order arrives as the comma string the Input node's shape uses —
    # which both functions used to reject, so a permute filled in by hand reported
    # no shape at all. See test_the_movement_ops_read_what_the_panel_stores below.
    ("permute typed into the panel",
     "torch.permute", {"dims": "0,2,3,1"},
     lambda t: torch.permute(t, (0, 2, 3, 1))),
    ("transpose typed into the panel",
     "torch.transpose", {"dims": "1,3"},
     lambda t: torch.transpose(t, 1, 3)),
    ("flatten from a start dim",
     "nn.Flatten", {"start_dim": 2},
     lambda t: torch.nn.Flatten(start_dim=2)(t)),
    ("narrow — the slicing op",
     "torch.narrow", {"dim": 1, "start": 0, "length": 16},
     lambda t: torch.narrow(t, 1, 0, 16)),
    ("narrow on a spatial axis",
     "torch.narrow", {"dim": 2, "start": 8, "length": 16},
     lambda t: torch.narrow(t, 2, 8, 16)),
    ("narrow with a negative dim",
     "torch.narrow", {"dim": -1, "start": 4, "length": 8},
     lambda t: torch.narrow(t, -1, 4, 8)),
    ("interpolate by scale_factor",
     "interpolate", {"scale_factor": 2, "mode": "nearest"},
     lambda t: torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")),
    ("interpolate to an explicit size",
     "interpolate", {"size": [64, 64], "mode": "nearest"},
     lambda t: torch.nn.functional.interpolate(t, size=[64, 64], mode="nearest")),
]


@pytest.mark.parametrize("label,op,params,real", PARITY, ids=[c[0] for c in PARITY])
def test_shape_function_matches_torch(label, op, params, real):
    expected = list(real(_t()).shape)
    got = _reported(op, params)
    assert got == expected, (
        f"{op} with {params} on {INP}: registry says {got}, torch says {expected}"
    )


# ── multi-input ops ───────────────────────────────────────────────────────────

def test_stack_inserts_an_axis_rather_than_preserving_the_shape():
    # The old behaviour returned [1,64,32,32] here, dropping the stacked axis
    # entirely. Two inputs stacked at dim 0 must gain a leading 2.
    expected = list(torch.stack([_t(), _t()], dim=0).shape)
    got = _reported("torch.stack", {"dim": 0}, [list(INP), list(INP)])
    assert got == expected == [2] + INP


def test_stack_at_a_later_dim():
    expected = list(torch.stack([_t(), _t()], dim=1).shape)
    assert _reported("torch.stack", {"dim": 1}, [list(INP), list(INP)]) == expected


def test_stack_declines_when_inputs_disagree():
    assert _reported("torch.stack", {"dim": 0}, [list(INP), [1, 32, 32, 32]]) is None


def test_cat_concatenates_on_its_dim():
    expected = list(torch.cat([_t(), _t()], dim=1).shape)
    assert _reported("torch.cat", {"dim": 1}, [list(INP), list(INP)]) == expected


def test_where_broadcasts_its_operands():
    a, bshape = list(INP), [1, 64, 1, 1]
    expected = list(torch.where(torch.zeros(*a) > 0, torch.zeros(*a), torch.zeros(*bshape)).shape)
    assert _reported("torch.where", {}, [a, list(bshape), list(bshape)]) == expected


# ── ops that must DECLINE rather than guess ───────────────────────────────────

@pytest.mark.parametrize("op", ["torch.split", "torch.chunk"])
def test_split_and_chunk_decline_because_they_return_tuples(op):
    # A node carries exactly one output_shape and edges have no output index, so
    # a tuple-returning op cannot be represented. Declining sets shape_error;
    # returning the input shape (the old behaviour) was a silent falsehood.
    assert _reported(op, {"dim": 1}) is None
    assert _reported(op, {}) is None


@pytest.mark.parametrize("params", [
    {},                       # no target shape at all
    {"shape": ""},            # blank
    {"shape": "1,-1,-1"},     # two wildcards
    {"shape": "1,3,7"},       # element count does not divide
    {"shape": "nonsense"},
])
def test_reshape_declines_when_the_target_is_unusable(params):
    assert _reported("torch.reshape", params) is None


@pytest.mark.parametrize("params", [
    {"dim": 1, "start": 60, "length": 16},   # runs off the end: 60+16 > 64
    {"dim": 9, "start": 0, "length": 1},     # dim out of range
    {"dim": 1, "start": -1, "length": 4},    # negative start
    {"dim": 1, "start": 0, "length": 0},     # empty window
    {"dim": 1, "start": 0},                  # missing length
    {},
])
def test_narrow_declines_on_an_impossible_slice(params):
    assert _reported("torch.narrow", params) is None


def test_narrow_accepts_the_exact_boundary():
    # start + length == size is legal and must NOT be refused.
    assert _reported("torch.narrow", {"dim": 1, "start": 48, "length": 16}) == [1, 16, 32, 32]


# ── the op that used to be knowingly wrong ────────────────────────────────────

def test_sum_reduces_the_way_torch_reduces():
    """torch.sum used to report its input shape unchanged, and every downstream node
    inherited that lie. It was held there because the real reduction exposed a wrong
    aggregation axis in `message_passing_block`; that template now sums over
    neighbours, so the rule is the real one and is pinned as such.

    All three shapes of the rule are checked against a real torch call, because the
    reduction is only correct if it handles the axis, keepdim AND the full-reduction
    case — a passthrough happens to look right for none of them.
    """
    assert _reported("torch.sum", {"dim": 1}) == list(torch.sum(_t(), dim=1).shape) == [1, 32, 32]
    assert (
        _reported("torch.sum", {"dim": 1, "keepdim": True})
        == list(torch.sum(_t(), dim=1, keepdim=True).shape)
        == [1, 1, 32, 32]
    )
    assert _reported("torch.sum", {"dim": -1}) == list(torch.sum(_t(), dim=-1).shape) == [1, 64, 32]
    assert _reported("torch.sum", {}) == list(torch.sum(_t()).shape) == [], "full reduction"


def test_the_shape_preserving_ops_really_are_shape_preserving():
    # _shape_passthrough backs 50+ specs and is CORRECT for these; the parity
    # check has to confirm that too, or the test above looks arbitrary.
    for op, real in [
        ("nn.ReLU", torch.nn.ReLU()),
        ("nn.GELU", torch.nn.GELU()),
        ("nn.Dropout", torch.nn.Dropout()),
        ("torch.abs", torch.abs),
        ("torch.sqrt", torch.sqrt),
    ]:
        expected = list(real(_t()).shape)
        assert _reported(op, {}) == expected == INP, op


def test_transpose_param_name_does_not_match_torchs_signature():
    """The shape function is right; the emitted call would not be.

    `_shape_transpose` reads `params["dims"]` as a 2-list, but torch's signature is
    `torch.transpose(input, dim0, dim1)` with no `dims` keyword — so codegen's
    generic `torch.{op}(x, **params)` tail emits `torch.transpose(x, dims=[1, 3])`,
    which raises TypeError. Pinned here so the divergence is visible; the emission
    side is handled by the functional-arg mapping in code_generation.py.
    """
    assert _reported("torch.transpose", {"dims": [1, 3]}) == [1, 32, 32, 64]
    assert _reported("torch.transpose", {"dim0": 1, "dim1": 3}) is None
    with pytest.raises(TypeError):
        torch.transpose(_t(), dims=[1, 3])


def test_the_movement_ops_read_what_the_panel_stores():
    """permute/transpose take their axes from `dims`, in the form the panel writes.

    The palette field used to be called `dim` — read by nothing, since the shape
    function, codegen and torch itself all want `dims`, so a permute filled in by
    hand reported no shape and Export refused it. Renaming the field is only half
    of it: the field is free text, so the value arrives as the same comma string
    the Input node's shape uses, and rejecting that leaves the identical dead end.
    """
    assert _reported("torch.permute", {"dims": "0,2,3,1"}) == [1, 32, 32, 64]
    assert _reported("torch.transpose", {"dims": "1,3"}) == [1, 32, 32, 64]
    # An empty field is still "not set", and declining is right — verify turns that
    # into "needs dims" (tests/test_verify_functional_ops.py).
    assert _reported("torch.permute", {"dims": ""}) is None
    assert _reported("torch.transpose", {"dims": ""}) is None
    # A wrong order is still refused rather than guessed at.
    assert _reported("torch.permute", {"dims": "0,2,3"}) is None
    assert _reported("torch.permute", {"dims": "0,0,1,2"}) is None
    # The old key stays dead on purpose: honouring it here would make the shape
    # card disagree with verify and Export, which both refuse a node with no `dims`.
    assert _reported("torch.permute", {"dim": "0,2,3,1"}) is None


# ── the new ops in a real graph, not just in isolation ────────────────────────

def test_narrow_and_interpolate_propagate_through_a_chain():
    """A shape function that is right on its own is not yet useful.

    This walks the whole engine: a slice narrows the channel axis, the next layer
    inherits the narrowed shape, and interpolate then doubles the spatial dims. If
    an op declines, everything downstream of it loses its input shape and the
    canvas reports "Node input shape could not be resolved" on the NEXT node —
    which is a confusing place to see the symptom, hence this test.
    """
    from models.architecture_models import ArchitectureEdge, ArchitectureNode
    from core import shape_engine

    def node(node_id, op, kind="op", **params):
        return ArchitectureNode(node_id=node_id, kind=kind, op=op, params=params)

    def edge(edge_id, source, target):
        return ArchitectureEdge(
            edge_id=edge_id, source_node_id=source, source_port="y",
            target_node_id=target, target_port="x",
        )

    nodes = [
        node("relu", "nn.ReLU", kind="layer"),
        node("slice", "torch.narrow", dim=1, start=0, length=1),
        node("gelu", "nn.GELU", kind="layer"),
        node("up", "interpolate", scale_factor=2, mode="nearest"),
    ]
    edges = [edge("e1", "relu", "slice"), edge("e2", "slice", "gelu"), edge("e3", "gelu", "up")]

    result = shape_engine.infer(nodes, edges, {"relu": [1, 3, 224, 224]})

    assert result["slice"].output_shape == [1, 1, 224, 224]
    assert result["slice"].shape_error is None
    # The narrowed shape is inherited, not the original.
    assert result["gelu"].output_shape == [1, 1, 224, 224]
    assert result["up"].output_shape == [1, 1, 448, 448]


def test_a_declining_op_does_not_silently_pass_a_wrong_shape_downstream():
    """The point of declining: downstream gets nothing rather than a lie."""
    from models.architecture_models import ArchitectureEdge, ArchitectureNode
    from core import shape_engine

    nodes = [
        ArchitectureNode(node_id="relu", kind="layer", op="nn.ReLU", params={}),
        # No `shape`, so reshape cannot know its output.
        ArchitectureNode(node_id="rs", kind="op", op="torch.reshape", params={}),
        ArchitectureNode(node_id="gelu", kind="layer", op="nn.GELU", params={}),
    ]
    edges = [
        ArchitectureEdge(edge_id="e1", source_node_id="relu", source_port="y",
                         target_node_id="rs", target_port="x"),
        ArchitectureEdge(edge_id="e2", source_node_id="rs", source_port="y",
                         target_node_id="gelu", target_port="x"),
    ]
    result = shape_engine.infer(nodes, edges, {"relu": [1, 3, 8, 8]})

    assert result["rs"].output_shape is None
    assert result["rs"].shape_error is not None
    # Crucially NOT [1, 3, 8, 8] — the old passthrough would have handed the
    # unchanged shape to gelu as though a reshape had been accounted for.
    assert result["gelu"].output_shape != [1, 3, 8, 8]


def test_the_inference_engine_reduces_sum_the_same_way_the_registry_does():
    """The registry rule above was right; a second plane shadowed it.

    `/api/models/infer-shapes` runs core.shape_inference, and that module routes
    `sum` through _ANALYTIC_MERGE_OPS before the registry rule is ever consulted —
    the set exists because a residual `sum` of two or more tensors has no torch.nn
    class and has to be applied directly. With a SINGLE input that route returned
    inputs[0] unchanged, which is the exact passthrough an older note records as
    fixed. Live, sum(dim=1) of a [1,3,224,224] input still answered [1,3,224,224]
    where torch answers [1,224,224].

    So this asserts the thing that actually matters: the two planes agree, and both
    agree with torch. Pinning only one plane is what let the bug survive its own fix.
    """
    for params in ({"dim": 1}, {"dim": 1, "keepdim": True}, {"dim": -1}, {"dim": 0}):
        real = list(torch.sum(_t(), **params).shape)
        engine = list(_apply_analytic_merge("sum", [_t()], params).shape)
        registry = _reported("torch.sum", params)
        assert engine == real, f"inference engine disagrees with torch for sum({params})"
        assert registry == real, f"registry disagrees with torch for sum({params})"

    # Full reduction, no dim: torch collapses to a scalar.
    assert list(_apply_analytic_merge("sum", [_t()], {}).shape) == list(torch.sum(_t()).shape) == []


def test_a_multi_input_sum_is_still_an_elementwise_residual():
    """The counterweight. The merge route exists for residual adds, and making the
    single-input case reduce must not turn a two-branch residual into a reduction."""
    a, b = _t(), _t()
    merged = _apply_analytic_merge("sum", [a, b], {"dim": 1})

    assert list(merged.shape) == list(INP), "a residual sum must preserve shape"
    assert torch.equal(merged, a + b)


def test_the_builder_reads_a_typed_parameter_and_leaves_a_functional_one_alone():
    """A kernel the canvas stores as text has to reach nn.Conv2d as a tuple.

    "3,3" is what a typed tuple field stores, and nn.Conv2d rejects the string
    outright ("type must be tuple of ints, but got str") — so
    /api/models/infer-shapes reported a layer error for a Conv2d that validate
    called fine and torch builds happily. The second half is the counterweight, and
    the reason that reading sits BELOW the functional branch rather than above it:
    TorchFunctionalLayer parses a comma-separated `dim` off the string itself, so
    normalising that one first turns mean(dim=(1,2)) into a reduction to a scalar.
    """
    conv = build_layer("Conv2d", {"in_channels": 3, "out_channels": 8, "kernel_size": "3,3"})
    assert conv.kernel_size == (3, 3)

    reduced = build_layer("torch.mean", {"dim": "1,2"})(_t())
    assert list(reduced.shape) == list(torch.mean(_t(), dim=(1, 2)).shape)

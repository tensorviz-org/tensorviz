"""The seed must read past a node that cannot answer — and stop at one that can (#196).

A canonical block has no Input node, so its input shape is inferred from the layer at
the entry. Several blocks open on `nn.Identity` — the GNN layers, the DenseNet blocks —
which says nothing about a shape, so the seed fell back to the expectation's default and
the first REAL layer refused it:

    nn.Linear(in_features=64) handed [1, 256]
      -> mat1 and mat2 shapes cannot be multiplied (1x256 and 64x64)
    nn.BatchNorm2d(num_features=64) handed [1, 3, 224, 224]
      -> running_mean should contain 3 elements not 64

The rule is "walk past ops that are SHAPE-NEUTRAL", and the distinction matters: the
first attempt walked past any node whose answer equalled the default, which is a
different rule and a wrong one. A `Conv2d(in_channels=3)` answers [1, 3, 224, 224] — the
default — and skipping it took a later Conv2d's channel count instead. Measured: 198
blocks clean fell to 137. Ask whether the node HAS a rule, not whether its answer is
surprising.
"""


from services.canonical_template_registry import CanonicalTemplateRegistry  # noqa: E402


def _seed(tid):
    registry = CanonicalTemplateRegistry()
    nodes, edges, entry, _exit, _meta = registry.instantiate(tid, tid, {})
    return registry.seed_input_shape(
        template_id=tid, nodes=nodes, entry_node_id=entry, edges=edges
    )


def test_it_reads_through_an_identity_to_the_layer_that_can_answer():
    # gat_layer opens on nn.Identity; the first sizing layer is Linear(in_features=64).
    assert _seed("gat_layer") == [1, 64]
    assert _seed("gcn_layer") == [1, 64]


def test_it_reads_through_to_a_convolution_channel_count():
    # densenet_transition's first sizing layer is Conv2d(in_channels=128).
    assert _seed("densenet_transition")[:2] == [1, 128]


def test_it_does_not_walk_past_a_node_that_answered_with_the_default():
    """The regression the first attempt caused, pinned.

    aspp_block's entry is Conv2d(in_channels=3), whose honest answer IS the default.
    Walking past it because the answer 'looks like' silence seeded the block from a
    later Conv2d and broke 61 blocks that had been clean.
    """
    assert _seed("aspp_block") == [1, 3, 224, 224]


def test_a_block_that_already_answered_at_its_entry_is_unchanged():
    assert _seed("vit_block") == [1, 16, 64]

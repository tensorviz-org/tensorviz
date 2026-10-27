"""The registry offers only blocks the code generator can actually build.

A block whose operations the generator has no entry for is not a template, it is a
picture: it draws onto the canvas and then cannot be turned into a `.py`, exported, or
run. Eighty-five were being offered — the graph-neural-network family and friends, whose
`neighbor_sampler` needs an edge index a generated `forward(x)` has no parameter for.
"""


from services.code_generation import generator_supports_op  # noqa: E402
from services.canonical_template_registry import CanonicalTemplateRegistry  # noqa: E402


def test_every_offered_block_uses_operations_the_generator_knows():
    """The oracle is the GENERATOR, not the op registry.

    The generator accepts an op from either source: the registry, or `_CUSTOM_OP_CLASS`,
    for which it emits a shape-preserving nn.Module into the exported file. Asking
    `spec_for` alone wrongly condemned `rotary_attention_block` and
    `latent_diffusion_block` — and, in the catalogue, F5-TTS.
    """
    registry = CanonicalTemplateRegistry()
    exempt = {"input", "output", "module"}
    unknown = {}
    for contract in registry.list_contracts():
        nodes, _e, _entry, _exit, _meta = registry.instantiate(
            contract.template_id, contract.template_id, {}
        )
        for node in nodes:
            op = str(getattr(node, "op", "") or "")
            if op.split(".")[-1].lower() in exempt:
                continue
            if not generator_supports_op(op):
                unknown.setdefault(contract.template_id, set()).add(op)
    assert not unknown, f"offered blocks using unknown operations: {unknown}"


def test_the_unbuildable_ones_are_still_there_to_be_fixed():
    # Removed from what is OFFERED, not deleted. The definitions stay so that teaching the
    # generator an operation brings its blocks back with nothing here to update.
    registry = CanonicalTemplateRegistry()
    assert len(registry.all_contracts()) > len(registry.list_contracts())


def test_the_graph_network_family_is_not_offered():
    # The clearest case, and the reason the rule exists: `neighbor_sampler` needs an edge
    # index, and a generated `forward(x)` has nowhere to put one.
    offered = {c.template_id for c in CanonicalTemplateRegistry().list_contracts()}
    assert "gnn_mean_1hop_block" not in offered
    assert "gnn_attention_3hop_block" not in offered


def test_the_ordinary_blocks_are_still_offered():
    # The rule must not have quietly emptied the catalogue.
    offered = {c.template_id for c in CanonicalTemplateRegistry().list_contracts()}
    for expected in ("vit_block", "resnet_bottleneck_block", "unet_block", "vgg_block"):
        assert expected in offered, expected
    assert len(offered) > 200


def test_buildable_is_computed_once():
    registry = CanonicalTemplateRegistry()
    first = registry.buildable_ids()
    assert registry.buildable_ids() is first

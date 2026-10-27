"""Regression tests for module placeholder expansion in TemplateService.

These guard the fix where every root-level moduleNode of a popular model must
resolve to a ModuleDefinition with non-empty internals (no dangling, empty
module boxes on the canvas), and where repeated stacks (transformer
encoder/decoder, ResNet stages) carry display-only repeat_count metadata.
"""


import pytest


from services.template_service import TemplateService  # noqa: E402


POPULAR_MODEL_IDS = [
    "bert_base",
    "gpt2_small",
    "t5_base",
    "resnet50",
    "vgg16",
    "inception_v3",
    "mobilenet_v2",
    "efficientnet_b0",
    "stylegan2",
    "biggan",
    "progressive_gan",
    "elmo",
    "fasttext",
    "word2vec",
]


@pytest.fixture(scope="module")
def service():
    return TemplateService()


@pytest.mark.parametrize("model_id", POPULAR_MODEL_IDS)
def test_no_dangling_module_nodes(service, model_id):
    """Every root-level moduleNode must have a modules[] entry with internals."""
    graph = service.instantiate_popular_model(model_id, {})

    dangling = []
    for node in graph.root_nodes:
        if node.type != "moduleNode":
            continue
        module_def = graph.modules.get(node.id)
        if module_def is None or not module_def.internal_nodes:
            dangling.append(node.id)

    assert dangling == [], (
        f"{model_id} has dangling (empty) module nodes: {dangling}"
    )


@pytest.mark.parametrize(
    "model_id,module_id,min_count,label",
    [
        ("bert_base", "encoder_stack", 12, "TransformerEncoderLayer"),
        ("gpt2_small", "decoder_stack", 12, "TransformerDecoderLayer"),
        ("resnet50", "layer1", 3, "Bottleneck"),
        ("resnet50", "layer3", 6, "Bottleneck"),
    ],
)
def test_repeated_stacks_carry_repeat_count(service, model_id, module_id, min_count, label):
    """Transformer stacks and ResNet stages expose repeat_count > 1 metadata,
    on BOTH the ModuleDefinition.parameters and the root node's data.parameters,
    while still materializing the full set of internal nodes."""
    graph = service.instantiate_popular_model(model_id, {})

    module_def = graph.modules.get(module_id)
    assert module_def is not None, f"{model_id}: module {module_id} missing"

    # ModuleDefinition carries the repeat metadata (survives .dict()).
    assert module_def.parameters.get("repeat_count") == min_count
    assert module_def.parameters.get("repeat_count") > 1
    assert module_def.parameters.get("repeat_block_label") == label

    # Root placeholder node mirrors the metadata for the frontend.
    root_node = next(n for n in graph.root_nodes if n.id == module_id)
    assert root_node.data.parameters.get("repeat_count") == min_count
    assert root_node.data.parameters.get("repeat_block_label") == label

    # repeat_count is display-only: internals must be FULLY materialized,
    # not reduced to a single block. A ResNet bottleneck = 9 leaf nodes,
    # so blocks*N layers; a transformer stack = exactly num_layers leaves.
    assert len(module_def.internal_nodes) >= min_count


def test_resnet_stage_materializes_all_blocks(service):
    """ResNet layer3 (6 blocks * 9 leaves/bottleneck) must keep all 54 leaves
    even though repeat_count says 6 — proves we don't collapse to one block."""
    graph = service.instantiate_popular_model("resnet50", {})
    module_def = graph.modules["layer3"]
    assert module_def.parameters["repeat_count"] == 6
    assert len(module_def.internal_nodes) == 6 * 9


def test_module_dict_roundtrip_preserves_repeat_count(service):
    """repeat_count must survive serialization (.dict()) since the API
    response is JSON. JSON shape: graph.modules[mid].parameters.repeat_count
    and root node data.parameters.repeat_count."""
    graph = service.instantiate_popular_model("bert_base", {})
    payload = graph.dict()
    assert payload["modules"]["encoder_stack"]["parameters"]["repeat_count"] == 12
    root = next(n for n in payload["root_nodes"] if n["id"] == "encoder_stack")
    assert root["data"]["parameters"]["repeat_count"] == 12

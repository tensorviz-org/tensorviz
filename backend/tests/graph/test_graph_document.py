

from services.graph_document import (
    dump_graph_document,
    graph_document_v2,
    model_graph_from_document,
)


def test_legacy_model_graph_converts_without_losing_reopen_payload():
    legacy = {
        "nodes": [
            {
                "id": "conv-1",
                "type": "layerNode",
                "position": {"x": 12.5, "y": 42},
                "selected": True,
                "data": {
                    "layerType": {"id": "Conv2d", "name": "Conv2d"},
                    "parameters": {"in_channels": 3, "out_channels": 16},
                    "collapsed": True,
                },
            }
        ],
        "edges": [],
    }

    document = graph_document_v2(legacy, graph_id="g1", name="CNN", revision=7)
    encoded = dump_graph_document(document)

    assert encoded["schemaVersion"] == 2
    assert encoded["semantic"]["graph_id"] == "g1"
    assert encoded["semantic"]["revision"] == 7
    assert encoded["semantic"]["nodes"][0]["op"] == "Conv2d"
    assert encoded["semantic"]["nodes"][0]["params"]["out_channels"] == 16
    assert encoded["semantic"]["metadata"]["migration"]["legacyPayload"] == legacy
    assert encoded["view"]["node_positions"]["conv-1"] == {"x": 12.5, "y": 42.0}
    assert encoded["view"]["selection"] == ["conv-1"]
    assert encoded["view"]["collapsed_modules"] == ["conv-1"]
    assert model_graph_from_document(document) == legacy


def test_legacy_conversion_prefers_canonical_layer_name_over_palette_id():
    legacy = {
        "nodes": [
            {
                "id": "linear-1",
                "position": {"x": 0, "y": 0},
                "data": {
                    "layerType": {"id": "linear", "name": "nn.Linear"},
                    "parameters": {},
                },
            }
        ],
        "edges": [],
    }

    document = graph_document_v2(legacy, graph_id="g", name="MLP", revision=1)

    assert document.semantic.nodes[0].op == "nn.Linear"


def test_v2_document_is_validated_and_stamped_with_record_identity():
    payload = {
        "schemaVersion": 2,
        "semantic": {
            "graph_id": "stale",
            "name": "Stale",
            "revision": 1,
            "nodes": [],
            "edges": [],
            "operations": [{"op": "validate"}],
        },
        "view": {
            "graph_id": "stale",
            "node_positions": {},
            "collapsed_modules": [],
            "selection": [],
            "layout": {"strategy": "dagre"},
        },
    }

    document = graph_document_v2(payload, graph_id="g2", name="Live", revision=3)

    assert document.schemaVersion == 2
    assert document.semantic.graph_id == "g2"
    assert document.semantic.name == "Live"
    assert document.semantic.revision == 3
    assert document.semantic.operations == [{"op": "validate"}]
    assert document.view.graph_id == "g2"

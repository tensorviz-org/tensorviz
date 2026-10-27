"""Regression test for the flat ShapeInferenceService Embedding rule.

The old formula computed [seq_len, embedding_dim] (dropping the batch dim and
producing the wrong rank). nn.Embedding actually appends embedding_dim as a new
trailing dimension: (N, L) token indices -> (N, L, embedding_dim).
"""



from core.shape_inference import ShapeInferenceService  # noqa: E402
from models.graph_models import Graph  # noqa: E402


def _node(node_id, layer_id, params, node_type="layerNode"):
    return {
        "id": node_id,
        "type": node_type,
        "position": {"x": 0, "y": 0},
        "data": {"layerType": {"id": layer_id, "name": layer_id, "parameters": []}, "parameters": params},
    }


def test_flat_embedding_appends_embedding_dim():
    graph = {
        "nodes": [
            _node("in", "input", {"shape": "1,1024"}, "inputNode"),
            _node("emb", "embedding", {"num_embeddings": 50257, "embedding_dim": 768}),
        ],
        "edges": [{"id": "e1", "source": "in", "target": "emb"}],
    }
    result = ShapeInferenceService().infer_shapes(Graph(**graph), [1, 1024])
    assert result["shapes"]["emb"] == [1, 1024, 768]

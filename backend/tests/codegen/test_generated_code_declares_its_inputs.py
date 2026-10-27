"""Generated code must say what to call its forward with (#196 / FAIL-10).

The trace worker runs the model. For a model whose `forward` takes several
arguments it has to know how many and of what dtype, and it was guessing: a
candidate pool that caps at THREE arguments. F5-TTS takes four — text, noisy
audio, conditioning, timestep — so it died with

    TypeError: F5Tts.forward() missing 1 required positional argument: 'in_time'

and every one of its 290 nodes was then reported "was not reached", which reads
as 286 defects in the user's model rather than one in ours.

The generator already knew all four: it writes exactly those tensors into the
`__main__` example, `randint` included, because floats crash an Embedding. It
simply never said so where the tracer could read it. `# tensorviz: input=` is
that place, and `parse_input_spec` already accepts a Python expression for the
dtype case.

The web app never hit this because it never RUNS a model — it infers shapes
analytically, where a four-input graph is unremarkable. Only the IDE traces.
"""


import torch  # noqa: E402

from services.code_generation import CodeGenerationService  # noqa: E402
from services.ingest.service import parse_input_spec  # noqa: E402


def _node(node_id, name, params=None, op=None):
    return {
        "id": node_id,
        "type": "inputNode" if name == "Input" else "outputNode" if name == "Output" else "layerNode",
        "position": {"x": 0, "y": 0},
        "data": {"layerType": {"id": op or name.lower(), "name": name, "parameters": []},
                 "parameters": params or {}},
    }


def _declaration(nodes, edges, shape=(1, 8)):
    from models.graph_models import Edge, Node

    service = CodeGenerationService()
    return service.input_declaration(
        [Node(**n) for n in nodes],
        [Edge(**e) for e in edges],
        list(shape),
    )


def test_a_multi_input_model_declares_every_argument():
    nodes = [
        _node("in_a", "Input", {"shape": "1,16"}),
        _node("in_b", "Input", {"shape": "1,32"}),
        _node("in_c", "Input", {"shape": "1,4"}),
        _node("in_d", "Input", {"shape": "1,2"}),
        _node("out", "Output"),
    ]
    declaration = _declaration(nodes, [])
    assert declaration.startswith("# tensorviz: input="), declaration
    args, kwargs = parse_input_spec(declaration.split("=", 1)[1])
    assert kwargs == {}
    # Four inputs must produce FOUR tensors. Three is the bug: the candidate pool
    # capped there and the fourth argument was simply never passed.
    assert len(args) == 4, declaration
    assert [tuple(t.shape) for t in args] == [(1, 16), (1, 32), (1, 4), (1, 2)]


def test_an_embedding_input_is_declared_as_integers():
    """floats crash nn.Embedding with an error about dtype, not about the input."""
    nodes = [
        _node("ids", "Input", {"shape": "1,16"}),
        _node("emb", "Embedding", {"num_embeddings": 257, "embedding_dim": 8}, op="embedding"),
        _node("out", "Output"),
    ]
    edges = [{"id": "e1", "source": "ids", "target": "emb"}, {"id": "e2", "source": "emb", "target": "out"}]
    declaration = _declaration(nodes, edges)
    args, _ = parse_input_spec(declaration.split("=", 1)[1])
    assert len(args) == 1
    assert args[0].dtype in (torch.int64, torch.int32), args[0].dtype


def test_a_single_input_model_still_declares_one():
    nodes = [_node("x", "Input", {"shape": "1,3,8,8"}), _node("out", "Output")]
    args, _ = parse_input_spec(_declaration(nodes, []).split("=", 1)[1])
    assert len(args) == 1
    assert tuple(args[0].shape) == (1, 3, 8, 8)


def test_the_declaration_uses_the_plain_form_so_BOTH_tiers_can_read_it():
    """Two tiers read `# tensorviz: input=`, and only one can evaluate Python.

    The trace worker imports torch and accepts either form. The parser deliberately
    does not — `code_parser.declared_input_shape` reads the plain comma form only,
    "which is exactly what this tier promises not to import".

    The first version of this emitted the expression form unconditionally, so every
    generated file's declaration was invisible to the parser. It fell back to its
    [1, 3, 224, 224] default, and the editor then reported "input shape is assumed,
    not stated — not checked" and skipped validation entirely. A green tick covering
    nothing is worse than a red one.
    """
    from utils.code_parser import declared_input_shape

    nodes = [
        _node("in_a", "Input", {"shape": "1,16,256"}),
        _node("relu", "ReLU"),
        _node("out", "Output"),
    ]
    declaration = _declaration(nodes, [])
    assert declaration == "# tensorviz: input=1,16,256", declaration
    # and the parser tier — the one that cannot evaluate Python — can read it
    assert declared_input_shape(declaration) == [1, 16, 256]


def test_an_embedding_still_falls_back_to_the_expression_form():
    """A shape alone cannot say `randint`, and floats crash an Embedding. The worker
    gets it exactly right; the parser falls back to its own inference, which is the
    honest trade rather than a wrong dtype."""
    from utils.code_parser import declared_input_shape

    nodes = [
        _node("ids", "Input", {"shape": "1,16"}),
        _node("emb", "Embedding", {"num_embeddings": 257, "embedding_dim": 8}, op="embedding"),
        _node("out", "Output"),
    ]
    edges = [{"id": "e1", "source": "ids", "target": "emb"}, {"id": "e2", "source": "emb", "target": "out"}]
    declaration = _declaration(nodes, edges)
    assert "randint" in declaration, declaration
    assert declared_input_shape(declaration) is None

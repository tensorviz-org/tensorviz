"""Whether a node can be validated, decided before anything runs (step 5.9).

Each test name states one rule. The rule they share: a refusal must arrive on the
card, not after the button, and must name the declaration that would lift it.
"""

from research.validatable import verdict_for


def test_a_module_taking_a_config_is_offered_the_config_this_project_declares():
    # The finding nanoGPT produced: four of its five modules take `config`, and the
    # index has ALREADY read GPTConfig's defaults for its own card. The information
    # was never missing — it had no path from the config card to the constructor.
    verdict = verdict_for(
        "model", "GPT", {"_ctor_required": ["config"]}, configurations=[{"label": "GPTConfig"}]
    )
    assert verdict.state == "needs"
    assert "GPTConfig" in verdict.detail
    assert verdict.remedy == 'tensorviz.init("GPT", config="GPTConfig")'


def test_several_configurations_are_listed_rather_than_one_being_guessed():
    verdict = verdict_for(
        "model",
        "GPT",
        {"_ctor_required": ["config"]},
        configurations=[{"label": "GPTConfig"}, {"label": "TinyConfig"}],
    )
    assert "GPTConfig, TinyConfig" in verdict.detail


def test_a_constructor_wanting_plain_values_names_them_all():
    # nanoGPT's LayerNorm(ndim, bias). No config to offer; the honest answer is
    # which parameters have no value, not a generic "could not be constructed".
    verdict = verdict_for("module", "LayerNorm", {"_ctor_required": ["ndim", "bias"]})
    assert "`ndim`, `bias`" in verdict.detail
    assert verdict.remedy.startswith('tensorviz.init("LayerNorm"')


def test_a_declaration_satisfies_the_requirement_it_names():
    verdict = verdict_for(
        "model",
        "GPT",
        {"_ctor_required": ["config"], "_init_args": {"config": "GPTConfig"}, "_in_shape": "1x1024"},
    )
    assert verdict.state == "ready"
    assert verdict.detail == ""


def test_a_constructible_model_with_no_example_input_says_so_separately():
    # Two different missing things, two different sentences. Collapsing them into
    # one "cannot validate" would send the user to fix the wrong half.
    verdict = verdict_for("model", "Net", {})
    assert verdict.state == "needs"
    assert "No example input" in verdict.detail
    assert verdict.remedy == 'tensorviz.input("Net", shape="?")'


def test_the_offered_input_shape_is_a_blank_and_not_a_plausible_guess():
    # The defect that started this: an image shape printed with the authority of a
    # measurement, on a language model. A remedy that suggests one is the same lie
    # one step earlier.
    verdict = verdict_for("model", "GPT", {})
    assert "224" not in verdict.remedy


def test_a_layer_that_states_its_own_width_is_not_asked_to_declare_one():
    # Conv2d(3, ...) already says what goes in. Demanding a declaration for what
    # the code states would be a refusal nobody deserves.
    verdict = verdict_for("model", "Net", {"_in_shape": "1x3x224x224"})
    assert verdict.state == "ready"


def test_a_declared_input_counts_as_an_input():
    verdict = verdict_for("model", "GPT", {"_input_declared": {"shape": "1x1024"}})
    assert verdict.state == "ready"


def test_a_node_with_no_forward_pass_is_unsupported_rather_than_broken():
    # Silence on a dataset card would read as "this failed". It did not fail; the
    # question does not apply to it.
    for kind in ("dataset", "configuration", "file", "trainer"):
        verdict = verdict_for(kind, "Thing", {})
        assert verdict.state == "unsupported", kind
        assert verdict.detail


def test_the_verdict_survives_the_wire_as_three_plain_strings():
    assert set(verdict_for("model", "Net", {}).as_dict()) == {"state", "detail", "remedy"}


def test_the_input_comment_counts_as_a_declared_input():
    """`# tensorviz: input=` is the authority a user actually writes.

    It is rung 2 of the precedence ladder and shipped long before the SDK existed,
    but the project graph only ever looked for `tensorviz.input()` — so
    `examples/tensorviz-test/models/tiny_transformer.py`, whose first line after the
    docstring declares its input exactly as the docs say, showed "Cannot validate:
    nothing here says what shape to run through it" and offered a declaration for
    something already declared.
    """
    from research.indexer import index_project

    sources = {
        "net.py": (
            '"""A model."""\n'
            "# tensorviz: input=torch.randint(1, 16, (1, 10))\n"
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class Net(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.embed = nn.Embedding(16, 32)\n"
        ),
        "train.py": "from net import Net\n\nif __name__ == '__main__':\n    Net()\n",
    }
    graph = index_project(sources).experiments[0].graph
    net = next(node for node in graph.nodes if node.label == "Net")
    assert verdict_for(net.kind, net.label, net.params).state == "ready"


def test_a_model_with_no_input_anywhere_still_says_so():
    """The counterweight. The verdict exists to refuse honestly, and a rule that
    answers `ready` for everything is the defect it was built to end."""
    from research.indexer import index_project

    sources = {
        "net.py": (
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class Net(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.act = nn.ReLU()\n"
        ),
        "train.py": "from net import Net\n\nif __name__ == '__main__':\n    Net()\n",
    }
    graph = index_project(sources).experiments[0].graph
    net = next(node for node in graph.nodes if node.label == "Net")
    assert verdict_for(net.kind, net.label, net.params).state == "needs"

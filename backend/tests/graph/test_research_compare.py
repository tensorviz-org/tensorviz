"""Comparing two experiments — the question a researcher asks constantly.

Answered from two present-tense graphs, so it needs no history and nothing on disk.
The tests use the repository's own example project, because two experiments that
genuinely share a model and genuinely differ is the case worth checking, and a
hand-built pair would only test what the author already believed.
"""

import os



from research.compare import compare  # noqa: E402
from research.indexer import index_project  # noqa: E402

SHARED_MODEL = "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n"


def config(**values):
    fields = "\n".join(f"    {name}: int = {value}" for name, value in values.items())
    return f"from dataclasses import dataclass\n\n\n@dataclass\nclass RunConfig:\n{fields}\n"


def two_experiments(left_config, right_config, right_extra=""):
    sources = {
        "models/net.py": SHARED_MODEL,
        "cfg_a.py": left_config,
        "cfg_b.py": right_config,
        "train_a.py": (
            "from cfg_a import RunConfig\n"
            "from models.net import Net\n"
            "\n"
            "if __name__ == '__main__':\n    Net()\n"
        ),
        "train_b.py": (
            "from cfg_b import RunConfig\n"
            "from models.net import Net\n"
            f"{right_extra}"
            "\n"
            "if __name__ == '__main__':\n    Net()\n"
        ),
    }
    index = index_project(sources)
    left = next(e for e in index.experiments if e.entrypoint == "train_a.py")
    right = next(e for e in index.experiments if e.entrypoint == "train_b.py")
    return left, right


def test_a_changed_hyperparameter_is_the_headline():
    """`n_layer: 12 -> 4` is the answer nine times out of ten."""
    left, right = two_experiments(config(n_layer=12, lr=3), config(n_layer=4, lr=3))
    diff = compare(left, right)

    changed = [s for s in diff.settings if s.name == "n_layer"]
    assert len(changed) == 1
    assert (changed[0].left, changed[0].right) == (12, 4)
    assert changed[0].only_one_side is False


def test_a_setting_both_share_is_not_reported():
    left, right = two_experiments(config(n_layer=12, lr=3), config(n_layer=4, lr=3))
    assert not [s for s in compare(left, right).settings if s.name == "lr"]


def test_a_setting_only_one_side_declares_is_flagged_as_such():
    """Existing in one run and not the other differs from two values disagreeing."""
    left, right = two_experiments(config(n_layer=12), config(n_layer=12, warmup=100))
    warmup = next(s for s in compare(left, right).settings if s.name == "warmup")
    assert warmup.only_one_side is True
    assert warmup.left is None
    assert warmup.right == 100


def test_a_shared_definition_is_recognised_across_two_graphs():
    """The node ids differ by experiment; the definition id is what matches."""
    left, right = two_experiments(config(n_layer=1), config(n_layer=2))
    diff = compare(left, right)
    assert "models/net.py::Net" in diff.shared
    assert not [n for n in diff.nodes if n.label == "Net"]


def test_something_only_one_experiment_has_is_reported_with_its_side():
    left, right = two_experiments(
        config(n_layer=1),
        config(n_layer=1),
        right_extra="\n\ndef evaluate(model):\n    return 0\n",
    )
    diff = compare(left, right)
    evaluate = next(n for n in diff.nodes if n.label == "evaluate")
    assert evaluate.side == "right"
    assert evaluate.kind == "evaluator"


def test_the_comparison_is_symmetric_and_names_no_baseline():
    """Neither experiment is "the original"; nothing here decides which is right."""
    left, right = two_experiments(config(n_layer=12), config(n_layer=4))
    forward, backward = compare(left, right), compare(right, left)

    assert forward.left == backward.right
    a = next(s for s in forward.settings if s.name == "n_layer")
    b = next(s for s in backward.settings if s.name == "n_layer")
    assert (a.left, a.right) == (b.right, b.left)


def test_two_identical_experiments_report_nothing():
    left, right = two_experiments(config(n_layer=1), config(n_layer=1))
    diff = compare(left, right)
    assert diff.is_empty()
    assert diff.shared, "identical is not the same as unrelated"


def test_comparing_unrelated_experiments_says_so_by_sharing_nothing():
    sources = {
        "a.py": "import torch.nn as nn\n\n\nclass A(nn.Module):\n    pass\n\n\nif __name__ == '__main__':\n    A()\n",
        "b.py": "import torch.nn as nn\n\n\nclass B(nn.Module):\n    pass\n\n\nif __name__ == '__main__':\n    B()\n",
    }
    index = index_project(sources)
    left = next(e for e in index.experiments if e.entrypoint == "a.py")
    right = next(e for e in index.experiments if e.entrypoint == "b.py")
    diff = compare(left, right)

    assert diff.shared == []
    assert {n.side for n in diff.nodes} == {"left", "right"}


def test_the_example_project_compares_its_own_experiments():
    """The end-to-end case: two real experiments sharing a real model."""
    root = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "..", "..", "examples", "tensorviz-research"
    )
    assert os.path.isdir(root), f"shipped example is missing: {root}"

    sources = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", "sdk", ".vscode"}]
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(base, name)
                with open(path, encoding="utf-8") as handle:
                    sources[os.path.relpath(path, root).replace(os.sep, "/")] = handle.read()

    index = index_project(sources)
    train = next(e for e in index.experiments if e.entrypoint == "train.py")
    sweep = next(e for e in index.experiments if e.entrypoint == "sweep.py")
    lora = next(e for e in index.experiments if e.entrypoint == "finetune_lora.py")

    # train.py and sweep.py are the same pipeline driven two ways.
    assert "models/tiny_lm.py::TinyLM" in compare(train, sweep).shared
    # The LoRA finetune shares nothing with them; it is different work.
    assert compare(train, lora).shared == []

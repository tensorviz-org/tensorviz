"""The indexer gate: sources in, experiments out, nothing executed.

The fixture at `fixtures/research_project/` is deliberately awkward — two entrypoints
sharing one model, one declared and one discovered, a file nobody imports, and a file
that does not parse. Each test pins a rule that would otherwise regress silently.
"""

import os

import pytest


from research import identity  # noqa: E402
from research.indexer import analyse_file, index_project  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures", "research_project")


def read_fixture(prefix=""):
    """`{rel: source}` for the fixture, optionally nested under a prefix."""
    sources = {}
    for base, dirs, files in os.walk(FIXTURE):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            rel = os.path.relpath(path, FIXTURE).replace(os.sep, "/")
            with open(path, encoding="utf-8") as handle:
                sources[f"{prefix}{rel}" if prefix else rel] = handle.read()
    return sources


@pytest.fixture(scope="module")
def index():
    return index_project(read_fixture())


# ── discovery ───────────────────────────────────────────────────────────────────


def test_two_entrypoints_give_two_experiments(index):
    entrypoints = sorted(e.entrypoint for e in index.experiments)
    assert entrypoints == ["train_a.py", "train_b.py"]
    assert len({e.experiment_id for e in index.experiments}) == 2


def test_declared_experiment_owns_its_identity(index):
    declared = next(e for e in index.experiments if e.entrypoint == "train_b.py")
    assert declared.label == "wide-net"
    assert declared.status.declared is True
    assert declared.experiment_id == identity.declared_experiment_id("wide-net")


def test_main_block_experiment_is_active_but_not_declared(index):
    inferred = next(e for e in index.experiments if e.entrypoint == "train_a.py")
    assert inferred.status.state == "active"
    assert inferred.status.declared is False


def test_filename_alone_is_a_candidate_not_an_assertion():
    """`run_sweep.py` looks like an entrypoint but proves nothing — the user confirms.

    It must still look like code: a file of nothing but literal assignments is
    configuration whatever it is called, and is not offered at all.
    """
    result = index_project({"run_sweep.py": "import os\n\n\ndef go():\n    return os.getcwd()\n"})
    assert [e.status.state for e in result.experiments] == ["candidate"]


# ── nothing disappears ──────────────────────────────────────────────────────────


def test_unparseable_file_is_a_node_with_its_error_not_a_hole(index):
    broken = [u for u in index.unreached if u.file == "broken.py"]
    assert broken and broken[0].error
    assert "(line" in broken[0].error


def test_file_nobody_imports_is_listed_as_unreached(index):
    assert "notes.py" in {u.file for u in index.unreached}


def test_every_closure_file_appears_as_a_file_node(index):
    experiment = next(e for e in index.experiments if e.entrypoint == "train_a.py")
    file_labels = {n.label for n in experiment.graph.nodes if n.kind == "file"}
    assert file_labels == set(experiment.closure)


# ── shared definitions ──────────────────────────────────────────────────────────


def test_shared_class_is_one_definition_per_experiment_with_distinct_nodes(index):
    definition = identity.definition_id("models/net.py", "Net")
    node_ids = set()
    for experiment in index.experiments:
        matches = experiment.graph.dependents_of_definition(definition)
        assert len(matches) == 1, f"{experiment.entrypoint} should use Net exactly once"
        node_ids.update(matches)
    assert len(node_ids) == 2, "two experiments must not share one node id"


# ── classification ──────────────────────────────────────────────────────────────


def test_classification_is_explainable(index):
    """Every classified node carries the sentence the canvas shows on hover."""
    experiment = next(e for e in index.experiments if e.entrypoint == "train_a.py")
    classified = [n for n in experiment.graph.nodes if n.kind not in ("file",)]
    assert classified
    assert all(n.params.get("_reason") for n in classified)


def test_model_and_dataset_are_recognised_by_base_class(index):
    experiment = next(e for e in index.experiments if e.entrypoint == "train_a.py")
    kinds = {n.label: n.kind for n in experiment.graph.nodes}
    assert kinds["Net"] == "model"
    assert kinds["ToyDataset"] == "dataset"


def test_trainer_is_recognised_by_backward_then_step(index):
    experiment = next(e for e in index.experiments if e.entrypoint == "train_a.py")
    kinds = {n.label: n.kind for n in experiment.graph.nodes}
    assert kinds["train"] == "trainer"
    assert kinds["evaluate"] == "evaluator"
    assert kinds["load_config"] == "configuration"


def test_usage_matching_sees_library_classes_the_user_never_subclassed():
    """The rule a definition-only recogniser misses: modern repos *instantiate*.

    Without this the whole file is one `generic` box and the product looks broken on
    any HuggingFace/TRL codebase.
    """
    source = (
        "from datasets import load_dataset\n"
        "from peft import LoraConfig\n"
        "from trl import SFTTrainer\n"
        "\n"
        "ds = load_dataset('imdb')\n"
        "cfg = LoraConfig(r=16)\n"
        "trainer = SFTTrainer(model=None, train_dataset=ds, peft_config=cfg)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    trainer.train()\n"
    )
    result = index_project({"sft.py": source})
    graph = result.experiments[0].graph
    kinds = {n.label: n.kind for n in graph.nodes}

    assert kinds.get("imdb") == "dataset", "load_dataset('imdb') must be a dataset node"
    assert kinds.get("cfg") == "configuration"
    assert kinds.get("trainer") == "trainer"

    by_id = {n.id: n for n in graph.nodes}
    flows = {
        (by_id[e.from_node_id].label, by_id[e.to_node_id].label, e.relationship)
        for e in graph.edges
    }
    assert ("imdb", "trainer", "feeds") in flows, "train_dataset= is a dataflow edge"
    assert ("cfg", "trainer", "runs_on") in flows


# ── SDK ─────────────────────────────────────────────────────────────────────────


def test_sdk_nodes_and_edges_carry_declared_provenance(index):
    experiment = next(e for e in index.experiments if e.entrypoint == "train_b.py")
    declared = {n.label: n for n in experiment.graph.nodes if n.provenance == "declared"}
    assert {"raw-logs", "tokenise"} <= set(declared)
    assert declared["raw-logs"].producer == "tensorviz.sdk"

    by_id = {n.id: n for n in experiment.graph.nodes}
    sdk_edges = {
        (by_id[e.from_node_id].label, by_id[e.to_node_id].label)
        for e in experiment.graph.edges
        if e.provenance == "declared" and e.relationship == "feeds"
    }
    assert ("raw-logs", "tokenise") in sdk_edges


def test_a_computed_sdk_argument_is_ignored_not_guessed():
    """The indexer reads the call; it never runs it. A variable label is unreadable."""
    source = (
        "import tensorviz\n"
        "name = 'computed'\n"
        "tensorviz.node('dataset', name)\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )
    result = index_project({"main.py": source})
    labels = {n.label for n in result.experiments[0].graph.nodes}
    assert "computed" not in labels


# ── derived project roots ───────────────────────────────────────────────────────


def test_project_root_is_derived_from_import_resolution():
    """A project nested in a bigger workspace resolves against its own directory."""
    sources = read_fixture(prefix="packages/alpha/")
    result = index_project(sources)
    assert result.roots == ["packages/alpha"]
    for experiment in result.experiments:
        assert experiment.root == "packages/alpha"
        assert all(f.startswith("packages/alpha/") for f in experiment.closure)


def test_two_nested_projects_separate_without_configuration():
    """The monorepo case: no marker files, no settings, just import evidence."""
    sources = {}
    sources.update(read_fixture(prefix="packages/alpha/"))
    sources.update(read_fixture(prefix="packages/beta/"))

    result = index_project(sources)
    assert result.roots == ["packages/alpha", "packages/beta"]

    for experiment in result.experiments:
        prefix = experiment.root + "/"
        assert all(f.startswith(prefix) for f in experiment.closure), (
            f"{experiment.entrypoint} leaked outside {experiment.root}"
        )
    assert len(result.experiments) == 4


def test_an_experiment_that_resolves_nothing_roots_at_the_workspace():
    """A script in a subfolder is in the project; it is not its own project.

    This asserted `scripts` until torchtitan and trl were indexed. A file whose
    imports are all relative or all third party resolves no prefix, so the root fell
    back to whatever directory the file sat in — and every such directory became a
    separate project: `.github/scripts`, `trl/experimental/bco`, 37 roots in one
    repository. `scripts/` is a folder inside a project, and with nothing to say
    otherwise the workspace is the project.
    """
    result = index_project({"scripts/solo.py": "if __name__ == '__main__':\n    pass\n"})
    assert result.experiments[0].root == ""


def test_a_genuinely_separate_project_still_gets_its_own_root():
    """The counterweight: erring shallow must not cost a monorepo its boundary.

    Two sibling projects, each resolving its own imports against its own directory.
    Nothing is declared; the prefix that made the import resolve is the evidence.
    """
    sources = {
        "packages/alpha/train.py": "import data.toy\n\nif __name__ == '__main__':\n    data.toy\n",
        "packages/alpha/data/toy.py": "X = 1\n",
        "packages/beta/train.py": "import data.toy\n\nif __name__ == '__main__':\n    data.toy\n",
        "packages/beta/data/toy.py": "X = 2\n",
    }
    roots = {e.entrypoint: e.root for e in index_project(sources).experiments}
    assert roots["packages/alpha/train.py"] == "packages/alpha"
    assert roots["packages/beta/train.py"] == "packages/beta"


# ── determinism and purity ──────────────────────────────────────────────────────


def test_indexing_is_deterministic(index):
    again = index_project(read_fixture())
    assert again.digest() == index.digest()


def test_analyse_file_never_raises_on_bad_syntax():
    facts = analyse_file("broken.py", "def oops(\n")
    assert facts.error and not facts.definitions


def test_truncation_is_carried_not_hidden():
    result = index_project({"a.py": "X = 1\n"}, truncated=True)
    assert result.truncated is True


# ── entrypoint discovery beyond the __main__ guard ──────────────────────────────
#
# Every case below was found by indexing real repositories (nanoGPT and
# examples/tensorviz-research) rather than invented, and each one was wrong before.


def test_a_plain_script_with_no_main_guard_is_an_active_experiment():
    """nanoGPT's train.py has no guard; it was invisible as a mere candidate."""
    sources = {
        "models/net.py": "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n",
        "sweep.py": (
            "from models.net import Net\n"
            "\n"
            "for depth in (2, 4):\n"
            "    model = Net()\n"
            "    print(depth)\n"
        ),
    }
    result = index_project(sources)
    sweep = next(e for e in result.experiments if e.entrypoint == "sweep.py")
    assert sweep.status.state == "active"
    assert "models/net.py" in sweep.closure


def test_a_library_module_is_not_an_experiment_however_it_is_named():
    """`training/trainer.py` matched `train.*` and became a junk experiment."""
    sources = {
        "training/trainer.py": "class Trainer:\n    pass\n",
        "run.py": "from training.trainer import Trainer\n\nTrainer()\n",
    }
    result = index_project(sources)
    assert [e.entrypoint for e in result.experiments] == ["run.py"]


def test_a_file_another_file_imports_is_a_module_not_a_candidate():
    sources = {
        "evaluation/evaluate.py": "def evaluate(model):\n    return 0\n",
        "train.py": "from evaluation.evaluate import evaluate\n\nif __name__ == '__main__':\n    evaluate(None)\n",
    }
    result = index_project(sources)
    assert [e.entrypoint for e in result.experiments] == ["train.py"]


def test_a_file_of_literal_assignments_is_configuration_not_an_experiment():
    """nanoGPT's config/ is seven files named train_*.py and eval_*.py."""
    sources = {
        "config/train_gpt2.py": "batch_size = 12\nblock_size = 1024\nwandb_log = True\n",
        "config/eval_gpt2.py": "init_from = 'gpt2'\neval_only = True\n",
    }
    result = index_project(sources)
    assert result.experiments == []
    assert {u.file for u in result.unreached} == set(sources)


# ── the inferred spine ──────────────────────────────────────────────────────────


def test_the_spine_picks_the_outermost_model_not_the_first_leaf():
    """A composed model's first nn.Module is a leaf; the reader means the top."""
    sources = {
        "models/parts.py": (
            "import torch.nn as nn\n\n\n"
            "class Attention(nn.Module):\n    pass\n\n\n"
            "class Block(nn.Module):\n    pass\n"
        ),
        "models/net.py": (
            "import torch.nn as nn\n\nfrom models.parts import Block\n\n\n"
            "class TinyLM(nn.Module):\n    pass\n"
        ),
        "data/toy.py": (
            "from torch.utils.data import Dataset\n\n\n"
            "class Corpus(Dataset):\n    pass\n"
        ),
        "train.py": (
            "from data.toy import Corpus\n"
            "from models.net import TinyLM\n"
            "\n"
            "if __name__ == '__main__':\n    TinyLM()\n"
        ),
    }
    graph = index_project(sources).experiments[0].graph
    by_id = {n.id: n for n in graph.nodes}
    feeds = {
        by_id[e.to_node_id].label
        for e in graph.edges
        if e.relationship == "feeds" and by_id[e.from_node_id].label == "Corpus"
    }
    assert "TinyLM" in feeds
    assert "Attention" not in feeds


def test_the_spine_never_contradicts_an_edge_read_from_the_source():
    """A guess must not argue with what the code actually says."""
    source = (
        "from datasets import load_dataset\n"
        "from trl import SFTTrainer\n"
        "\n"
        "ds = load_dataset('imdb')\n"
        "model = None\n"
        "trainer = SFTTrainer(model=model, train_dataset=ds)\n"
        "\n"
        "if __name__ == '__main__':\n    trainer.train()\n"
    )
    graph = index_project({"sft.py": source}).experiments[0].graph
    by_id = {n.id: n for n in graph.nodes}
    pairs = [
        (by_id[e.from_node_id].label, by_id[e.to_node_id].label, e.relationship, e.provenance)
        for e in graph.edges
    ]
    assert ("imdb", "trainer", "feeds", "static") in pairs
    inferred_between = [
        p for p in pairs if p[3] == "inferred" and {p[0], p[1]} == {"imdb", "trainer"}
    ]
    assert not inferred_between, f"a guess duplicated a static edge: {inferred_between}"


def test_writing_a_checkpoint_does_not_make_something_a_trainer():
    """`save_checkpoint` is a utility the trainer calls, not a second trainer."""
    sources = {
        "ckpt.py": "import torch\n\n\ndef save_checkpoint(model):\n    torch.save(model, 'x.pt')\n",
        "train.py": "from ckpt import save_checkpoint\n\nif __name__ == '__main__':\n    save_checkpoint(None)\n",
    }
    graph = index_project(sources).experiments[0].graph
    kinds = {n.label: n.kind for n in graph.nodes}
    assert kinds["save_checkpoint"] == "checkpoint"


# ── found by indexing nanoGPT ───────────────────────────────────────────────────


def test_a_module_level_training_loop_is_a_trainer():
    """nanoGPT runs its loop at top level, so the graph had no trainer at all."""
    source = (
        "from models.net import Net\n"
        "\n"
        "model = Net()\n"
        "optimizer = None\n"
        "while True:\n"
        "    loss = model(None)\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
    )
    sources = {"models/net.py": "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n", "train.py": source}
    graph = index_project(sources).experiments[0].graph
    trainers = [n for n in graph.nodes if n.kind == "trainer"]
    assert trainers, "a file whose module body trains is itself the trainer"
    assert "module body" in trainers[0].params["_reason"]


def test_a_chained_backward_call_still_counts_as_training():
    """`scaler.scale(loss).backward()` loses its receiver and reads as `backward`."""
    source = (
        "from models.net import Net\n"
        "\n"
        "model = Net()\n"
        "scaler = None\n"
        "for _ in range(2):\n"
        "    loss = model(None)\n"
        "    scaler.scale(loss).backward()\n"
        "    scaler.step(None)\n"
    )
    sources = {"models/net.py": "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n", "train.py": source}
    graph = index_project(sources).experiments[0].graph
    assert any(n.kind == "trainer" for n in graph.nodes)


def test_a_local_inside_a_class_method_is_not_a_project_node():
    """nanoGPT's `model_hf` is an internal of GPT.from_pretrained, not a model."""
    sources = {
        "model.py": (
            "import torch.nn as nn\n"
            "from transformers import GPT2LMHeadModel\n"
            "\n"
            "\n"
            "class GPT(nn.Module):\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, kind):\n"
            "        model_hf = GPT2LMHeadModel.from_pretrained(kind)\n"
            "        return model_hf\n"
        ),
        "train.py": "from model import GPT\n\nif __name__ == '__main__':\n    GPT()\n",
    }
    graph = index_project(sources).experiments[0].graph
    labels = {n.label for n in graph.nodes}
    assert "GPT" in labels
    assert "model_hf" not in labels, "a class's private local leaked onto the canvas"


def test_a_batch_helper_is_recognised_as_data():
    graph = index_project(
        {"train.py": "def get_batch(split):\n    return 1, 2\n\nif __name__ == '__main__':\n    get_batch('train')\n"}
    ).experiments[0].graph
    kinds = {n.label: n.kind for n in graph.nodes}
    assert kinds["get_batch"] == "dataset"


def test_the_spine_picks_the_model_nothing_else_is_built_from():
    """model.py defines LayerNorm .. GPT in order; the first is a leaf."""
    sources = {
        "model.py": (
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class LayerNorm(nn.Module):\n    pass\n"
            "\n"
            "\n"
            "class Block(nn.Module):\n"
            "    def __init__(self):\n        self.ln = LayerNorm()\n"
            "\n"
            "\n"
            "class GPT(nn.Module):\n"
            "    def __init__(self):\n        self.block = Block()\n"
        ),
        "data.py": "from torch.utils.data import Dataset\n\n\nclass Corpus(Dataset):\n    pass\n",
        "train.py": (
            "from data import Corpus\n"
            "from model import GPT\n"
            "\n"
            "if __name__ == '__main__':\n    GPT()\n"
        ),
    }
    graph = index_project(sources).experiments[0].graph
    by_id = {n.id: n for n in graph.nodes}
    fed = {
        by_id[e.to_node_id].label
        for e in graph.edges
        if e.relationship == "feeds" and by_id[e.from_node_id].label == "Corpus"
    }
    assert "GPT" in fed
    assert "LayerNorm" not in fed


# ── found by indexing VoxCPM ────────────────────────────────────────────────────


SRC_LAYOUT = {
    "src/pkg/__init__.py": "",
    "src/pkg/model.py": "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n",
    "src/pkg/training/__init__.py": "from .loop import Trainer\nfrom .data import Corpus\n",
    "src/pkg/training/loop.py": "class Trainer:\n    def step(self):\n        pass\n",
    "src/pkg/training/data.py": (
        "from torch.utils.data import Dataset\n\n\nclass Corpus(Dataset):\n    pass\n"
    ),
    "scripts/train.py": (
        "from pkg.model import Net\n"
        "from pkg.training import Trainer\n"
        "\n"
        "if __name__ == '__main__':\n    Net()\n"
    ),
}


def test_a_src_layout_package_resolves():
    """`src/` is a sys.path root because it holds a package and is not one.

    Without this a src-layout repository resolves nothing at all — VoxCPM's
    training script reached none of the model code it imports.
    """
    result = index_project(SRC_LAYOUT)
    experiment = next(e for e in result.experiments if e.entrypoint == "scripts/train.py")
    assert "src/pkg/model.py" in experiment.closure


def test_relative_imports_inside_a_package_are_followed():
    """`from .data import Corpus` is the norm inside a package, not an exception."""
    result = index_project(SRC_LAYOUT)
    experiment = next(e for e in result.experiments if e.entrypoint == "scripts/train.py")
    assert "src/pkg/training/data.py" in experiment.closure
    assert "src/pkg/training/loop.py" in experiment.closure


def test_the_project_root_spans_the_script_and_the_package():
    """scripts/ and src/ are both parts of one project; neither alone is it."""
    result = index_project(SRC_LAYOUT)
    experiment = next(e for e in result.experiments if e.entrypoint == "scripts/train.py")
    assert experiment.root == ""


def test_two_classes_sharing_a_name_are_told_apart():
    """VoxCPM keeps audio_vae.py beside audio_vae_v2.py, each defining Snake1d."""
    sources = {
        "m/__init__.py": "",
        "m/vae.py": "import torch.nn as nn\n\n\nclass Snake1d(nn.Module):\n    pass\n",
        "m/vae_v2.py": "import torch.nn as nn\n\n\nclass Snake1d(nn.Module):\n    pass\n",
        "train.py": (
            "from m.vae import Snake1d\n"
            "from m.vae_v2 import Snake1d as S2\n"
            "\n"
            "if __name__ == '__main__':\n    Snake1d()\n"
        ),
    }
    graph = index_project(sources).experiments[0].graph
    labels = sorted(n.label for n in graph.nodes if n.kind == "model")
    assert labels == ["Snake1d (vae)", "Snake1d (vae_v2)"], labels


def test_a_unique_label_is_left_alone():
    graph = index_project(
        {
            "m/__init__.py": "",
            "m/net.py": "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n",
            "train.py": "from m.net import Net\n\nif __name__ == '__main__':\n    Net()\n",
        }
    ).experiments[0].graph
    assert any(n.label == "Net" for n in graph.nodes)


# ── configuration is the thing a researcher actually came to read ───────────────


def test_a_dataclass_config_shows_its_hyperparameters():
    """nanoGPT's GPTConfig is seven fields. A card that only says "named like a
    configuration" tells the reader nothing the name did not already."""
    sources = {
        "model.py": (
            "from dataclasses import dataclass\n"
            "\n"
            "\n"
            "@dataclass\n"
            "class GPTConfig:\n"
            "    block_size: int = 1024\n"
            "    n_layer: int = 12\n"
            "    dropout: float = 0.0\n"
            "    bias: bool = True\n"
        ),
        "train.py": "from model import GPTConfig\n\nif __name__ == '__main__':\n    GPTConfig()\n",
    }
    graph = index_project(sources).experiments[0].graph
    config = next(n for n in graph.nodes if n.label == "GPTConfig")
    assert config.kind == "configuration"
    assert config.params["settings"] == {
        "block_size": 1024,
        "n_layer": 12,
        "dropout": 0.0,
        "bias": True,
    }
    assert "4 settings" in config.params["_reason"]


def test_a_plain_class_of_assignments_counts_too():
    sources = {
        "cfg.py": "class TrainConfig:\n    lr = 0.0003\n    epochs = 10\n",
        "train.py": "from cfg import TrainConfig\n\nif __name__ == '__main__':\n    TrainConfig()\n",
    }
    graph = index_project(sources).experiments[0].graph
    config = next(n for n in graph.nodes if n.label == "TrainConfig")
    assert config.params["settings"] == {"lr": 0.0003, "epochs": 10}


def test_argparse_defaults_are_settings():
    sources = {
        "train.py": (
            "import argparse\n"
            "\n"
            "\n"
            "def parse_args():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--lr', default=0.0003)\n"
            "    parser.add_argument('--batch-size', default=32)\n"
            "    return parser.parse_args()\n"
            "\n"
            "if __name__ == '__main__':\n    parse_args()\n"
        ),
    }
    graph = index_project(sources).experiments[0].graph
    config = next(n for n in graph.nodes if n.label == "parse_args")
    assert config.kind == "configuration"
    # The flag becomes the name a reader knows it by.
    assert config.params["settings"] == {"lr": 0.0003, "batch_size": 32}


def test_a_computed_default_is_left_out_rather_than_quoted():
    """`default=os.cpu_count()` is a value this cannot know without running the
    code; showing the expression text instead would be a different kind of lie."""
    sources = {
        "train.py": (
            "import argparse, os\n"
            "\n"
            "\n"
            "def parse_args():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--workers', default=os.cpu_count())\n"
            "    parser.add_argument('--lr', default=0.001)\n"
            "    return parser.parse_args()\n"
            "\n"
            "if __name__ == '__main__':\n    parse_args()\n"
        ),
    }
    graph = index_project(sources).experiments[0].graph
    settings = next(n for n in graph.nodes if n.label == "parse_args").params["settings"]
    assert settings["lr"] == 0.001
    assert settings["workers"] is None


def test_a_huge_config_is_capped_rather_than_flooding_the_card():
    from research.classify import MAX_SETTINGS

    fields = "\n".join(f"    field_{i}: int = {i}" for i in range(MAX_SETTINGS + 20))
    sources = {
        "cfg.py": f"from dataclasses import dataclass\n\n\n@dataclass\nclass BigConfig:\n{fields}\n",
        "train.py": "from cfg import BigConfig\n\nif __name__ == '__main__':\n    BigConfig()\n",
    }
    graph = index_project(sources).experiments[0].graph
    settings = next(n for n in graph.nodes if n.label == "BigConfig").params["settings"]
    assert len(settings) == MAX_SETTINGS


def test_a_model_class_is_not_reclassified_by_its_attributes():
    """A class with literal class attributes is not thereby a configuration."""
    sources = {
        "net.py": (
            "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    scale = 2\n"
        ),
        "train.py": "from net import Net\n\nif __name__ == '__main__':\n    Net()\n",
    }
    graph = index_project(sources).experiments[0].graph
    assert next(n for n in graph.nodes if n.label == "Net").kind == "model"


def test_a_variable_assigned_in_both_branches_is_one_node():
    """Found by indexing `state-spaces/mamba`, which failed outright.

    Its generation benchmark picks a tokenizer in an if/else, so the file holds
    two `tokenizer = AutoTokenizer.from_pretrained(...)` statements. Identities
    carry no line number on purpose — that is what lets a node keep its id across
    an edit that moves it — so both assignments produced the id
    `<file>::<usage>tokenizer`, and `validate_graph` refuses duplicate ids. The
    cost was not a drawing glitch: the whole repository raised
    GraphValidationError and nothing rendered at all.
    """
    sources = {
        "bench.py": (
            "import argparse\n"
            "from transformers import AutoTokenizer\n"
            "\n"
            "args = argparse.ArgumentParser().parse_args()\n"
            "if args.model_name.startswith('state-spaces'):\n"
            "    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')\n"
            "else:\n"
            "    tokenizer = AutoTokenizer.from_pretrained(args.model_name)\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    tokenizer\n"
        ),
    }
    graph = index_project(sources).experiments[0].graph

    tokenizers = [n for n in graph.nodes if n.definition_id.endswith("<usage>tokenizer")]
    assert len(tokenizers) == 1

    # And it is named for the variable, not for the model the FIRST branch names:
    # labelling it "EleutherAI/gpt-neox-20b" would assert a choice the code makes
    # at runtime, which is exactly what a static graph must not claim.
    assert tokenizers[0].label == "tokenizer"
    assert "_label_hint" not in tokenizers[0].params
    # The reader is told there is more than one line behind this node, so looking
    # for "the" assignment cannot silently find one of two.
    assert tokenizers[0].params["_assignments"] == 2


def test_a_single_assignment_still_gets_its_label_from_the_literal():
    """The counterweight: grouping must not cost the common case its name."""
    sources = {
        "bench.py": (
            "from transformers import AutoTokenizer\n"
            "\n"
            "tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    tokenizer\n"
        ),
    }
    graph = index_project(sources).experiments[0].graph
    node = next(n for n in graph.nodes if n.definition_id.endswith("<usage>tokenizer"))
    assert node.label == "EleutherAI/gpt-neox-20b"
    assert "_assignments" not in node.params


# ── what is an entrypoint, and what is merely a module ──────────────────────────
#
# Every test below came from indexing `pytorch/torchtitan` (596 files, 262
# "experiments") and `huggingface/trl` (353 files, 216). Neither number was a
# rendering problem: an experiment rail listing 262 entries cannot be read, and a
# reader has no way to tell which four of them are the repository's actual
# entrypoints.


def test_a_file_something_imports_is_a_module_however_much_it_does():
    """Module-level work does not make a library module an entrypoint.

    `logger = logging.getLogger(__name__)` is a call, and the rule that recognised
    nanoGPT's guard-less `train.py` counted it as "runs at import". 120 of
    torchtitan's library modules and 89 of trl's were entrypoints on that basis.
    Something importing the file settles it: the project USES this, it does not RUN
    it.
    """
    sources = {
        "lib/util.py": "import logging\n\nlogger = logging.getLogger(__name__)\n\n\ndef helper():\n    return 1\n",
        "train.py": "from lib.util import helper\n\nif __name__ == '__main__':\n    helper()\n",
    }
    entrypoints = {e.entrypoint for e in index_project(sources).experiments}
    assert entrypoints == {"train.py"}


def test_a_guard_in_an_imported_file_is_a_candidate_not_an_entrypoint():
    """`python -m pkg.module` is real, so a guard is not nothing — but it is not the
    project's entrypoint either, and a candidate says exactly that."""
    sources = {
        "lib/tool.py": "def go():\n    return 1\n\nif __name__ == '__main__':\n    go()\n",
        "train.py": "from lib.tool import go\n\nif __name__ == '__main__':\n    go()\n",
    }
    states = {e.entrypoint: e.status.state for e in index_project(sources).experiments}
    assert states == {"train.py": "active", "lib/tool.py": "candidate"}


def test_tests_are_not_experiments():
    """torchtitan has 111 files with `if __name__ == '__main__': unittest.main()`.

    What a test runner collects is not inferred — `pytest` collects `test_*.py` and
    `*_test.py`, and `conftest.py` is its own configuration file.
    """
    body = "if __name__ == '__main__':\n    pass\n"
    sources = {
        "tests/test_model.py": body,
        "tests/conftest.py": body,
        "pkg/model_test.py": body,
        "pkg/tests/helpers.py": body,
        "train.py": body,
    }
    entrypoints = {e.entrypoint for e in index_project(sources).experiments}
    assert entrypoints == {"train.py"}


def test_a_declaration_still_wins_over_the_test_convention():
    """The convention is evidence; a declaration is an assertion, and outranks it.

    Otherwise naming an experiment `test_my_idea.py` would make it permanently
    unreachable with no way to say otherwise.
    """
    sources = {
        "tests/test_my_idea.py": (
            "import tensorviz\n\ntensorviz.experiment('my-idea')\n"
        ),
    }
    experiments = index_project(sources).experiments
    assert [e.label for e in experiments] == ["my-idea"]
    assert experiments[0].status.declared is True


def test_repository_tooling_is_not_an_experiment():
    """`.github/scripts` and `.claude/skills/*/scripts` ship beside the source
    without being it. torchtitan offered four as experiments and three as separate
    *projects*."""
    body = "if __name__ == '__main__':\n    pass\n"
    sources = {".github/scripts/release.py": body, "train.py": body}
    entrypoints = {e.entrypoint for e in index_project(sources).experiments}
    assert entrypoints == {"train.py"}


def test_an_import_guard_is_not_a_program():
    """`try: import flash_attn` and `if TYPE_CHECKING:` describe the environment.

    Both used to count as "the module body does work", which is most of how a
    library module came to look like a script.
    """
    sources = {
        "lib/fast.py": (
            "from typing import TYPE_CHECKING\n"
            "\n"
            "try:\n"
            "    import flash_attn\n"
            "except ImportError:\n"
            "    flash_attn = None\n"
            "\n"
            "if TYPE_CHECKING:\n"
            "    import torch\n"
            "\n"
            "\n"
            "def attend():\n"
            "    return flash_attn\n"
        ),
        "other.py": "import lib.fast\n",
    }
    assert index_project(sources).experiments == []


def test_a_real_module_level_loop_is_still_an_entrypoint():
    """The counterweight. nanoGPT's `train.py` has no guard and is the whole point
    of the rung the tests above narrow."""
    sources = {
        "model.py": "class Net:\n    pass\n",
        "train.py": (
            "from model import Net\n"
            "\n"
            "net = Net()\n"
            "while True:\n"
            "    net\n"
        ),
    }
    experiments = index_project(sources).experiments
    assert [e.entrypoint for e in experiments] == ["train.py"]
    assert experiments[0].status.state == "active"


# ── where a project begins ──────────────────────────────────────────────────────


def test_a_file_inside_a_package_roots_above_the_package():
    """`trl/experimental/bco/bco_trainer.py` sits under three `__init__.py` files.

    It imports its siblings relatively, so it resolves no prefix at all — and the
    root used to fall back to the file's own folder, which is a package and cannot
    be a project. That gave trl 37 roots, one per package subdirectory.
    """
    sources = {
        "pkg/__init__.py": "",
        "pkg/sub/__init__.py": "",
        "pkg/sub/thing.py": "from .other import x\n\nwhile True:\n    x\n",
        "pkg/sub/other.py": "x = 1\n",
    }
    roots = {e.root for e in index_project(sources).experiments}
    assert roots == {""}


def test_an_entrypoint_with_no_local_imports_joins_the_project_around_it():
    """DeepSeek-V3 keeps all five files under `inference/`, and `convert.py` imports
    nothing local.

    It proves nothing about where its project begins, so the answer is what the rest
    of the repository proved. Falling back to the workspace instead split a
    five-file repository into two projects.
    """
    sources = {
        "inference/model.py": "class Net:\n    pass\n",
        "inference/generate.py": "from model import Net\n\nif __name__ == '__main__':\n    Net()\n",
        "inference/convert.py": "import os\n\nif __name__ == '__main__':\n    os.getcwd()\n",
    }
    roots = {e.entrypoint: e.root for e in index_project(sources).experiments}
    assert roots == {"inference/generate.py": "inference", "inference/convert.py": "inference"}


def test_a_src_layout_is_one_project_from_either_side():
    """VoxCPM is entered both from `app.py` and from `src/voxcpm/cli.py`.

    The two resolve at different prefixes and reach different files — the CLI only
    the timestamp helpers, the app only the model — so comparing closures directly
    says "two projects". What settles it is that the outer project reaches INTO
    `src/`: that is where this project keeps its code, not a project of its own.
    """
    sources = {
        "src/pkg/__init__.py": "",
        "src/pkg/core.py": "class Core:\n    pass\n",
        "src/pkg/cli.py": "from pkg.helpers import h\n\nif __name__ == '__main__':\n    h()\n",
        "src/pkg/helpers.py": "def h():\n    return 1\n",
        "app.py": "from pkg.core import Core\n\nif __name__ == '__main__':\n    Core()\n",
    }
    roots = {e.root for e in index_project(sources).experiments}
    assert roots == {""}


def test_a_nested_project_nothing_reaches_stays_its_own():
    """The counterweight, and it is a real repository: `cleanrl/ppo_trxl/` ships its
    own `pyproject.toml` and nothing outside it imports a line.

    Absorbing every nested root would erase exactly the case derived roots exist
    for.
    """
    sources = {
        "train.py": "import os\n\nif __name__ == '__main__':\n    os.getcwd()\n",
        "sub/project/__init__.py": "",
        "sub/project/net.py": "class Net:\n    pass\n",
        "sub/project/run.py": "from project.net import Net\n\nif __name__ == '__main__':\n    Net()\n",
    }
    roots = {e.entrypoint: e.root for e in index_project(sources).experiments}
    assert roots["sub/project/run.py"] == "sub"
    assert roots["train.py"] == ""


def test_root_derivation_does_not_depend_on_set_iteration_order():
    """Two roots of equal length must not be ordered by a hash.

    The absorb pass and the fallback that settles an evidence-free entrypoint both
    take the FIRST enclosing root they find, so ordering `{"aa/bb", "cc/dd"}` by
    length alone left the answer to `PYTHONHASHSEED`. A digest that changes between
    runs over identical source is the one defect the content-addressed cache cannot
    survive — it would re-upload and re-render a workspace nobody touched.
    """
    sources = {
        "aa/bb/net.py": "class Net:\n    pass\n",
        "aa/bb/run.py": "import net\n\nif __name__ == '__main__':\n    net.Net()\n",
        "cc/dd/net.py": "class Net:\n    pass\n",
        "cc/dd/run.py": "import net\n\nif __name__ == '__main__':\n    net.Net()\n",
        "solo.py": "import os\n\nif __name__ == '__main__':\n    os.getcwd()\n",
    }
    first = index_project(sources)
    assert first.digest() == index_project(dict(reversed(list(sources.items())))).digest()
    assert {e.entrypoint: e.root for e in first.experiments} == {
        "aa/bb/run.py": "aa/bb",
        "cc/dd/run.py": "cc/dd",
        "solo.py": "",
    }


# ── incremental re-index (step 3.1) ─────────────────────────────────────────────


def test_a_warm_index_equals_a_cold_one():
    """The cache is an optimisation, so the only thing that matters is that it
    cannot change an answer.

    Measured over the validation corpus, parsing is 88-93% of a whole-repo index —
    1.32s of torchtitan's 1.49s across 596 files. So a keystroke does not need a
    reverse-import index to recompute only the affected experiments; it needs to
    not re-parse the 595 files that did not change. Warm: 0.23s.
    """
    from research.admission import ParseCache

    sources = {
        "models/net.py": "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n",
        "data/toy.py": "def make_loader():\n    return []\n",
        "train.py": (
            "from models.net import Net\n"
            "from data.toy import make_loader\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    Net()\n"
            "    make_loader()\n"
        ),
    }
    cache = ParseCache()
    index_project(sources, parse_cache=cache)

    edited = dict(sources)
    edited["models/net.py"] = edited["models/net.py"] + "\n\nclass Other(nn.Module):\n    pass\n"

    warm = index_project(edited, parse_cache=cache)
    assert warm.digest() == index_project(edited).digest()
    # And it really did re-use the two files that did not change.
    assert cache.hits == 2


def test_the_cache_cannot_carry_a_cross_file_answer_between_indexes():
    """The trap, and it is not hypothetical.

    `imports_workspace` and `imported_by_others` are decided per index by a pass
    that WRITES to the parsed facts. Handing the same object back on the next index
    would leave a file believing it is imported after the only file importing it was
    deleted — so it would stay a module and stop being offered as an experiment,
    silently, until something evicted the entry.
    """
    from research.admission import ParseCache

    cache = ParseCache()
    both = {
        "lib.py": "def helper():\n    return 1\n\nif __name__ == '__main__':\n    helper()\n",
        "train.py": "from lib import helper\n\nif __name__ == '__main__':\n    helper()\n",
    }
    warm_up = index_project(both, parse_cache=cache)
    assert {e.entrypoint: e.status.state for e in warm_up.experiments} == {
        "train.py": "active",
        "lib.py": "candidate",
    }

    # The importer is gone. `lib.py` is nobody's module any more.
    alone = {"lib.py": both["lib.py"]}
    after = index_project(alone, parse_cache=cache)
    assert [(e.entrypoint, e.status.state) for e in after.experiments] == [("lib.py", "active")]
    assert after.digest() == index_project(alone).digest()


def test_two_identical_files_are_cached_apart():
    """Keyed on the path as well as the digest, unlike the SOURCE cache.

    The parsed facts carry the file's own path — its definitions are identified as
    `file::qualname` — so two byte-identical `__init__.py` files do not have
    identical facts, and sharing one entry would put one package's nodes in
    another's graph.
    """
    from research.admission import ParseCache

    body = "class Thing:\n    pass\n"
    sources = {
        "a/__init__.py": body,
        "b/__init__.py": body,
        "train.py": "import a\nimport b\n\nif __name__ == '__main__':\n    a, b\n",
    }
    cache = ParseCache()
    index = index_project(sources, parse_cache=cache)
    files = {node.source.file for node in index.experiments[0].graph.nodes if node.source}
    assert {"a/__init__.py", "b/__init__.py"} <= files


# ── what counts as a model (D13, D16) ───────────────────────────────────────────


def test_a_class_subclassing_a_local_model_is_a_model():
    """D13, found in `deepseek-ai/DeepSeek-V3`.

    It defines `Linear(nn.Module)` and then `ColumnParallelLinear(Linear)` and
    `RowParallelLinear(Linear)` — the building blocks of its attention — and both
    came out `generic`, "a class with no recognised role", because the rule matched
    the literal base name only. Every serious repository defines its own base class.
    """
    sources = {
        "model.py": (
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class Base(nn.Module):\n    pass\n"
            "\n"
            "\n"
            "class Middle(Base):\n    pass\n"
            "\n"
            "\n"
            "class Leaf(Middle):\n    pass\n"
        ),
        "train.py": "from model import Leaf\n\nif __name__ == '__main__':\n    Leaf()\n",
    }
    graph = index_project(sources).experiments[0].graph
    kinds = {node.label: node.kind for node in graph.nodes}
    # Iterated to a fixed point, because the chain is not always one link.
    assert kinds["Base"] == kinds["Middle"] == kinds["Leaf"] == "model"
    reason = next(n for n in graph.nodes if n.label == "Leaf").params["_reason"]
    assert "Middle" in reason


def test_subclassing_a_torch_layer_is_a_model():
    """D16, found in `openai/whisper`.

    It overrides three torch layers — `class Linear(nn.Linear)`, `LayerNorm`,
    `Conv1d` — to change one line of forward, and all three read as "a class with no
    recognised role" in a repository whose whole subject is a model. Anything out of
    `torch.nn` is a Module by construction.
    """
    sources = {
        "layers.py": (
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class Linear(nn.Linear):\n    pass\n"
            "\n"
            "\n"
            "class Norm(torch.nn.LayerNorm):\n    pass\n"
        ),
        "train.py": "from layers import Linear\n\nif __name__ == '__main__':\n    Linear()\n",
    }
    graph = index_project(sources).experiments[0].graph
    kinds = {node.label: node.kind for node in graph.nodes}
    assert kinds["Linear"] == "model"
    assert kinds["Norm"] == "model"


def test_a_parameter_subclass_is_not_a_model():
    """The counterweight. `nn.Parameter` lives in the same namespace and is not a
    Module, so the rule is a deny-list over that namespace rather than "anything
    called nn-something"."""
    sources = {
        "p.py": "import torch.nn as nn\n\n\nclass Weight(nn.Parameter):\n    pass\n",
        "train.py": "from p import Weight\n\nif __name__ == '__main__':\n    Weight()\n",
    }
    graph = index_project(sources).experiments[0].graph
    assert next(n for n in graph.nodes if n.label == "Weight").kind != "model"


def test_the_closure_runs_the_package_inits_python_would_run():
    """D15, found in `openai/whisper`: **zero models in an ASR repository**.

    Its entrypoint is `whisper/__main__.py` -> `.transcribe`, and every
    `from .model import Whisper` in the package sits inside `if TYPE_CHECKING:` — a
    typing-only import that never runs. The one that does run is in
    `whisper/__init__.py`, which the closure did not walk, so `model.py` was in no
    experiment's closure at all and its eleven `nn.Module` classes were absent.

    Importing `pkg.mod` executes `pkg/__init__.py` first. That is what makes a
    package a package, and a library's `__init__.py` is very often where it
    re-exports the things worth drawing.
    """
    sources = {
        "pkg/__init__.py": "from .model import Net\n",
        "pkg/model.py": "import torch.nn as nn\n\n\nclass Net(nn.Module):\n    pass\n",
        "pkg/run.py": "def go():\n    return 1\n",
        "pkg/__main__.py": "from .run import go\n\nif __name__ == '__main__':\n    go()\n",
    }
    experiment = index_project(sources).experiments[0]
    assert "pkg/__init__.py" in experiment.closure
    assert "pkg/model.py" in experiment.closure
    assert any(node.label == "Net" and node.kind == "model" for node in experiment.graph.nodes)

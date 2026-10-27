"""The external corpus is pinned without vendoring or executing it in unit tests."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import model_corpus as corpus
from services.ingest.execution_spec import validate_execution_spec


def test_every_recipe_is_explicit_and_pinned():
    recipes = json.loads(corpus.RECIPES.read_text())["recipes"]
    assert len({r["id"] for r in recipes}) == len(recipes)
    assert {r["repository"] for r in recipes} == {"tensorsssss", "Pytorch-UNet-master", "dinov2-main", "whisper-main"}
    for recipe in recipes:
        assert recipe["file"] in corpus.pinned_files(recipe["repository"])
        assert ("outputShape" in recipe) != ("errorTypes" in recipe)
        assert recipe["execution"]["device"] == "cpu"
        resolved = validate_execution_spec(recipe["execution"])
        assert all(resolved[key] == value for key, value in recipe["execution"].items())
        assert resolved["gradEnabled"] is False and resolved["cwd"] == "."


@pytest.fixture
def small_corpus(tmp_path, monkeypatch):
    original = tmp_path / "original"
    (original / "model").mkdir(parents=True)
    (original / "model/net.py").write_text("raise RuntimeError('must not execute')\n")
    (original / "model/unpinned.py").write_text("# never copy")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"model/net.py": hashlib.sha256((original / "model/net.py").read_bytes()).hexdigest()}))
    monkeypatch.setattr(corpus, "MANIFEST", manifest)
    return original, tmp_path / "copy"


def test_copy_reads_only_verified_bytes_and_never_imports(small_corpus):
    original, target = small_corpus
    assert corpus.verified_copy(original, "model", target) == 1
    assert (target / "net.py").read_bytes() == (original / "model/net.py").read_bytes()
    assert not (target / "unpinned.py").exists()
    with pytest.raises(ValueError, match="must be new"):
        corpus.verified_copy(original, "model", target)


def test_corrupted_source_refuses_before_execution(small_corpus):
    original, target = small_corpus
    (original / "model/net.py").write_text("# changed")
    with pytest.raises(ValueError, match="Pinned corpus differs"):
        corpus.verified_copy(original, "model", target)


def test_original_tree_cannot_be_used_as_a_destination(small_corpus):
    original, _ = small_corpus
    with pytest.raises(ValueError, match="outside"):
        corpus.verified_copy(original, "model", original / "new")


def test_symlink_escape_refuses(small_corpus, tmp_path):
    original, target = small_corpus
    file = original / "model/net.py"
    other = tmp_path / "outside.py"
    file.rename(other)
    file.symlink_to(other)
    with pytest.raises(ValueError, match="in-repository"):
        corpus.verified_copy(original, "model", target)


def test_reference_failures_keep_the_underlying_type_after_setup_preflight(tmp_path):
    (tmp_path / 'corpus_binding_model.py').write_text('''from torch import nn
class Model(nn.Module):
    def forward(self, *, tokens): raise AssertionError('must not execute')
''')
    result = corpus.run_one(tmp_path, {'id': 'missing-tokens', 'file': 'corpus_binding_model.py',
        'class': 'Model', 'execution': {'inputs': {'args': []}}, 'errorTypes': ['TypeError']})
    assert result['matched'] and result['underlyingErrorType'] == 'TypeError'
    assert result['setupDiagnostic']['names'] == ['tokens']
    assert result['calledModules'] == []


def test_whisper_random_factory_initializes_checkpoint_owned_state_reproducibly(monkeypatch):
    import runpy
    import sys
    from types import ModuleType, SimpleNamespace
    import torch

    class Whisper(torch.nn.Module):
        def __init__(self, dimensions):
            super().__init__()
            self.dimensions = dimensions
            self.decoder = torch.nn.Module()
            self.decoder.positional_embedding = torch.nn.Parameter(torch.full(
                (dimensions.n_text_ctx, dimensions.n_text_state), float('nan')))
            self.untouched = torch.nn.Parameter(torch.full((2,), 17.0))

    module = ModuleType('whisper.model')
    module.Whisper, module.ModelDimensions = Whisper, SimpleNamespace
    monkeypatch.setitem(sys.modules, 'whisper.model', module)
    factory = runpy.run_path(str(corpus.FACTORY))
    with torch.random.fork_rng():
        for name in ('make_model', 'make_tiny'):
            torch.manual_seed(7)
            first = factory[name]()
            torch.manual_seed(7)
            second = factory[name]()
            assert torch.isfinite(first.decoder.positional_embedding).all()
            assert torch.equal(first.decoder.positional_embedding, second.decoder.positional_embedding)
            assert torch.equal(first.untouched, torch.full((2,), 17.0))
            assert first.dimensions == second.dimensions


def test_expected_shape_cannot_hide_a_nonfinite_reference_output(tmp_path):
    (tmp_path / 'nonfinite_corpus_model.py').write_text('''import torch
class Model(torch.nn.Module):
    def forward(self, x): return x * float('nan')
''')
    result = corpus.run_one(tmp_path, dict(id='nonfinite', file='nonfinite_corpus_model.py',
        **{'class': 'Model'}, execution=dict(inputs=dict(args=[dict(shape=[1], dtype='float32', fill='ones')])),
        outputShape=[1]))
    assert result['outcome'] == 'completed' and result['outputShape'] == [1]
    assert result['outputFinite'] is False and result['matched'] is False

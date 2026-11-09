import pytest
from services.ingest import torch_cache


def test_older_torch_skips_only_absent_optional_package(monkeypatch):
    monkeypatch.setattr(torch_cache.importlib.util, 'find_spec',
                        lambda name: object() if name == 'torch._dynamo' else None)
    monkeypatch.setattr(torch_cache.importlib, 'import_module', lambda name: pytest.fail('absent module imported'))
    torch_cache.prepare_cache()


def test_present_cache_is_initialized_and_its_errors_propagate(monkeypatch):
    monkeypatch.setattr(torch_cache.importlib.util, 'find_spec', lambda name: object())
    seen = []
    monkeypatch.setattr(torch_cache.importlib, 'import_module', lambda name: seen.append(name))
    torch_cache.prepare_cache()
    assert seen == ['torch._dynamo.package']
    def broken(name):
        raise ModuleNotFoundError('unrelated dependency', name='unrelated')
    monkeypatch.setattr(torch_cache.importlib, 'import_module', broken)
    with pytest.raises(ModuleNotFoundError, match='unrelated dependency'):
        torch_cache.prepare_cache()


def test_missing_optional_parent_is_not_imported(monkeypatch):
    seen = []
    def find(name):
        seen.append(name)
        assert name == 'torch._dynamo'
        return None
    monkeypatch.setattr(torch_cache.importlib.util, 'find_spec', find)
    torch_cache.prepare_cache()
    assert seen == ['torch._dynamo']

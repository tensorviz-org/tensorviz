"""PR discovery must find dependent models in snapshots without running code."""
import hashlib

from tests.graph.test_ide_multifile_projection import client


def index(client, sources, **options):
    return client.post('/api/ide/review/index', json={"protocol_version": 1,
        "sources": [{"file": file, "source": source, "digest": hashlib.sha256(source.encode()).hexdigest()}
                    for file, source in sources.items()], **options})


def test_snapshot_discovers_local_inheritance_shared_dependencies_and_duplicate_names(client):
    sources = {
        "blocks.py": "from torch import nn\nclass Block(nn.Module):\n    def forward(self,x): return x\n",
        "a.py": "from blocks import Block\nraise RuntimeError('never execute')\nclass Net(Block): pass\n",
        "b.py": "from blocks import Block as B\nclass Net(B): pass\n",
        "helper.py": "def preprocess(x): return x\n",
        "c.py": "from torch import nn as layers\nfrom helper import preprocess\nclass Net(layers.Module):\n    def forward(self,x): return preprocess(x)\n",
    }
    result = index(client, sources)
    assert result.status_code == 200, result.text
    models = {(item['file'], item['cls']): item for item in result.json()['models']}
    assert set(models) == {('a.py','Net'), ('b.py','Net'), ('blocks.py','Block'), ('c.py','Net')}
    assert models['a.py','Net']['dependencies'] == ['a.py', 'blocks.py']
    assert models['c.py','Net']['dependencies'] == ['c.py', 'helper.py']


def test_missing_external_base_and_syntax_errors_are_visible_instead_of_no_models(client):
    result = index(client, {'broken.py': 'class ??', 'external.py':
        'from library import Model\nclass Net(Model):\n    def forward(self,x): return x\n'}, truncated=True).json()
    assert result['models'] == []
    assert {note['file'] for note in result['notes']} == {'broken.py', 'external.py'}
    assert any(note['detail'] == 'could not parse broken.py:1' for note in result['notes'])
    assert result['truncated']


def test_rename_fingerprint_ignores_name_but_not_model_body_or_import_context(client):
    source = 'from torch import nn\nclass Net(nn.Module):\n    def forward(self,x): return x + 1\n'
    fingerprint = lambda source: index(client, {'model.py': source}).json()['models'][0]['fingerprint']
    assert fingerprint(source) == fingerprint(source.replace('class Net', 'class Renamed'))
    assert fingerprint(source) != fingerprint(source.replace('+ 1', '+ 2'))
    assert fingerprint(source) != fingerprint('from other import helper\n' + source)


def test_empty_revision_supports_whole_model_additions_and_deletions(client):
    assert index(client, {}).json() == {'models': [], 'notes': [], 'truncated': False}


def test_review_shares_module_container_alias_and_fallback_discovery(client):
    sources = {'model.py': '''from torch import nn
class BlockChunk(nn.ModuleList):
    def forward(self,x): return x
class Base(nn.Module): pass
try:
    from optional_library import OptionalBlock
except ImportError:
    OptionalBlock = Base
class Fused(OptionalBlock): pass
'''}
    result = index(client, sources).json()
    assert {model['cls'] for model in result['models']} == {'BlockChunk', 'Base', 'Fused'}
    assert result['notes'] == []


def test_optional_external_imports_do_not_create_blanket_warnings(client):
    source = '''from torch import nn
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
class Net(nn.Module):
    def forward(self,x): return x
'''
    assert index(client, {'model.py': source}).json()['notes'] == []


def test_conditional_local_imports_remain_in_affected_model_dependencies(client):
    source = '''from torch import nn
if enabled:
    from optional_block import Block
class Net(nn.Module):
    def forward(self,x): return x
'''
    result = index(client, {'model.py': source, 'optional_block.py': 'class Block: pass'}).json()
    assert result['models'][0]['dependencies'] == ['model.py', 'optional_block.py']


def test_dynamic_discovery_note_names_the_actual_unresolved_construct(client):
    result = index(client, {'model.py': 'import importlib\nBlock = importlib.import_module(name)\n'}).json()
    assert len(result['notes']) == 1
    assert 'Line 2' in result['notes'][0]['detail']
    assert 'importlib.import_module()' in result['notes'][0]['detail']


def test_discovery_rejects_forged_source_digest_and_paths(client):
    result = index(client, {'../outside.py': 'pass'})
    assert result.status_code == 400
    result = client.post('/api/ide/review/index', json={'protocol_version': 1,
        'sources': [{'file': 'model.py', 'source': 'pass', 'digest': '0' * 64}]})
    assert result.status_code == 400
    assert result.json()['detail']['code'] == 'digest_mismatch'

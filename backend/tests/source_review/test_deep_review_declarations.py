"""Source declarations remain reviewable when runtime containers cannot be lowered."""
from copy import deepcopy
import hashlib

from services.ide.multifile import MultifileProjectionRequest
from services.ide.workspace_projection import project_workspace
from services.semantic.document import capture_document
from tests.source_review.test_semantic_constructor_edits import setup


SOURCE = '''import torch
from torch import nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
    def forward(self, x):
        return self.proj(x)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(2)])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
'''


def project(source, extra=None, cls='Model', review_only=True):
    files = {'model.py': source, **(extra or {})}
    return project_workspace(MultifileProjectionRequest.model_validate(dict(protocol_version=1,
        root=dict(file='model.py', cls=cls, source_digest=hashlib.sha256(source.encode()).hexdigest()),
        sources=[dict(file=f, source=s, digest=hashlib.sha256(s.encode()).hexdigest()) for f, s in files.items()],
        review_only=review_only)))


def test_deep_declarations_are_compared_without_running_or_expanding_the_model():
    source = SOURCE + '\nraise RuntimeError("must not execute review")\n'
    before = project(source)
    after = project(source.replace('bias=False', 'bias=True'))
    facts = before['document']['semantic']['metadata']['reviewDeclarations']
    assert facts['complete']
    key = next(key for key, fact in facts['entries'].items() if fact['label'] == 'Block.proj')
    fact = facts['entries'][key]
    assert fact['digest'] != after['document']['semantic']['metadata']['reviewDeclarations']['entries'][key]['digest']
    assert {field: before['spanmap'][key][field] for field in fact['span']} == fact['span']
    assert not any(node['node_id'].endswith('.proj') for node in before['document']['semantic']['nodes'])
    moved = project('\n' + source)['document']['semantic']['metadata']['reviewDeclarations']['entries'][key]
    assert moved['digest'] == fact['digest']
    assert moved['span']['start'][0] == fact['span']['start'][0] + 1


def document(capture, source):
    return capture_document(capture, name='Model', graph_id='g', model_file='model.py',
        source_digests={'model.py': hashlib.sha256(source.encode()).hexdigest()},
        source_graph={'nodes': [], 'edges': []}, source_spans={}, source_texts={'model.py': source})[0]


def test_exact_constructors_locate_every_dynamic_child_instance(tmp_path):
    capture, _ = setup(tmp_path, SOURCE)
    doc = document(capture, SOURCE)
    facts = doc.semantic.metadata['reviewDeclarations']['entries']
    key = next(key for key, fact in facts.items() if fact['label'] == 'Block.proj')
    calls = doc.semantic.metadata['sourceDeclarationCalls'][key]
    module_ids = {call.moduleId for call in capture.invocations if call.id in calls}
    assert [m.paths for m in capture.modules if m.id in module_ids] == [['blocks.0.proj'], ['blocks.1.proj']]
    assert len(calls) == 2
    assert all(call in {c.id for c in capture.invocations} for call in calls)
    # No constructor proof or a shared alias cannot specialize a captured child.
    for reason in ['unverified', 'multiple_bindings', 'shared']:
        invalid = deepcopy(capture)
        for module in invalid.modules:
            if module.id in module_ids:
                if reason == 'unverified': module.constructorClassVerified = False
                elif reason == 'multiple_bindings':
                    module.constructorBindingIds = []
                    module.constructorClassVerified = False
                else: module.paths.append(module.paths[0] + '_alias')
        assert not document(invalid, SOURCE).semantic.metadata['sourceDeclarationCalls'].get(key)


def test_declaration_evidence_refuses_duplicate_classes_writes_and_budget_overflow(monkeypatch):
    from services.ide import review_declarations as review
    ambiguous = SOURCE.replace('self.proj = nn.Linear(4, 4, bias=False)',
        'self.proj = nn.Linear(4, 4)\n        self.proj = nn.ReLU()')
    facts = review.review_declarations({'model.py': ambiguous})
    assert next(f for f in facts['entries'].values() if f['label'] == 'Block.proj')['ambiguous']
    duplicate = review.review_declarations({'model.py': SOURCE + '\nclass Block: pass\n'})
    assert not duplicate['complete']
    assert not any(f['label'] == 'Block.proj' for f in duplicate['entries'].values())
    monkeypatch.setattr(review, 'MAX_DECLARATIONS', 1)
    assert review.review_declarations({'model.py': SOURCE}) == {'entries': {}, 'complete': False}


def test_exact_constructor_rejects_stale_bytes_and_token_evidence(tmp_path):
    import pytest
    capture, _ = setup(tmp_path, SOURCE)
    with pytest.raises(ValueError, match='snapshots differ'):
        document(capture, SOURCE.replace('bias=False', 'bias=True'))
    for binding in capture.sourceBindings:
        if binding.kind == 'constructor':
            binding.anchors[0].tokenDigest = 'a' * 64
    assert not document(capture, SOURCE).semantic.metadata['sourceDeclarationCalls']


LITERAL_SOURCE = SOURCE.replace('self.proj = nn.Linear(4, 4, bias=False)',
    'self.proj = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.ReLU())')


def test_literal_container_records_exact_slots_and_surrounding_source(tmp_path):
    before = project(LITERAL_SOURCE)['document']['semantic']['metadata']['reviewDeclarations']['entries']
    after_source = LITERAL_SOURCE.replace('nn.ReLU(), nn.ReLU()', 'nn.GELU(), nn.ReLU()')
    after = project(after_source)['document']['semantic']['metadata']['reviewDeclarations']['entries']
    key = next(key for key, fact in before.items() if fact['label'] == 'Block.proj[1]')
    assert before[key]['digest'] != after[key]['digest']
    assert before[key]['contextDigest'] == after[key]['contextDigest']
    for changed in [LITERAL_SOURCE.replace('nn.ReLU(), nn.ReLU()', 'nn.GELU(), nn.ReLU(), nn.ReLU()'),
                    LITERAL_SOURCE.replace('nn.Linear(4, 4), nn.ReLU()', 'nn.ReLU(), nn.Linear(4, 4)')]:
        current = project(changed)['document']['semantic']['metadata']['reviewDeclarations']['entries']
        assert before[key]['contextDigest'] != current[key]['contextDigest']
    capture, _ = setup(tmp_path, LITERAL_SOURCE)
    doc = document(capture, LITERAL_SOURCE)
    assert len(doc.semantic.metadata['sourceDeclarationCalls'][key]) == 2


def test_shadowed_torch_container_does_not_prove_literal_slots():
    from services.ide.review_declarations import review_declarations
    source = LITERAL_SOURCE.replace('nn.Sequential', 'torch.nn.Sequential') + '\ntorch = fake_namespace\n'
    facts = review_declarations({'model.py': source})['entries']
    assert not any(fact.get('parent') for fact in facts.values())


NET = '''import torch
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.tag = nn.Identity(name="déjà")
        self.seq = nn.Sequential(nn.Linear(8, 8), nn.Identity(note="é"))
    def forward(self, x):
        return self.seq(self.tag(self.fc(x)))
'''


def location(span):
    return span['file'], tuple(span['start']), tuple(span['end'])


def test_live_source_projection_carries_no_review_declarations():
    # The live Canvas sync projects without review_only. A declaration key has
    # its layer's exact span, so the editor cursor offered a raw review ID beside
    # `fc`, and a static edit counted it as another instance of that layer.
    live = project(NET, cls='Net', review_only=False)
    assert not live['traced'] and 'fc' in live['spanmap']
    assert [key for key in live['spanmap'] if key.startswith('@declaration:')] == []
    assert 'reviewDeclarations' not in live['document']['semantic']['metadata']
    # Git review and source drafts still navigate to each declaration.
    review = project(NET, cls='Net')
    key = '@declaration:["model.py","Net","fc"]'
    assert location(review['spanmap'][key]) == location(review['document']['semantic']['metadata']['reviewDeclarations']['entries'][key]['span'])


def test_declaration_spans_count_lines_as_python_does():
    # str.splitlines() also breaks at form feeds, U+0085 and U+2028, so an AST
    # line number read an earlier line: columns came from the wrong text, and a
    # multibyte character there crashed the projection.
    from services.ide.review_declarations import review_declarations
    source = ('from torch import nn\n\x0c\nclass Net(nn.Module):\n    def __init__(self):\n'
              '        super().__init__()  # éééééé\n        self.fc = nn.Linear(8, 4)\n'
              '    def forward(self, x): return self.fc(x)\n')
    for variant in [source, source.replace('  # éééééé', ''), source.replace('\x0c', '# a b'),
                    source.replace('\x0c', '# a\x85b'), source.replace('\n', '\r\n'), source.replace('\n', '\r')]:
        fact = review_declarations({'m.py': variant})['entries']['@declaration:["m.py","Net","fc"]']
        assert (fact['span']['start'], fact['span']['end']) == ([6, 8], [6, 33]), repr(variant)
        assert (fact['constructor']['start'], fact['constructor']['end']) == ([6, 18], [6, 33]), repr(variant)
    assert '@declaration:["model.py","Net","fc"]' in project(source, cls='Net')['spanmap']


def test_declaration_spans_join_static_nodes_and_constructors_join_capture_anchors(tmp_path):
    # Git review pairs each static node with its declaration by exact location.
    # Both count UTF-16 code units, as VS Code does; a declaration counted in
    # another unit missed every statement with non-ASCII text, so one edit was
    # reported twice.
    review = project(NET, cls='Net')
    facts = review['document']['semantic']['metadata']['reviewDeclarations']['entries']
    assert {fact['label'] for fact in facts.values()} == {'Net.fc', 'Net.tag', 'Net.seq', 'Net.seq[0]', 'Net.seq[1]'}
    static = {location(span) for key, span in review['spanmap'].items() if not key.startswith('@')}
    assert [fact['label'] for fact in facts.values() if location(fact['span']) not in static] == []
    # Recorded constructor anchors count UTF-16 code units (str indexes here).
    line = NET.split('\n')[6]
    constructor = facts['@declaration:["model.py","Net","tag"]']['constructor']
    assert (constructor['start'], constructor['end']) == ([7, line.index('nn.Identity')], [7, len(line)])
    source = SOURCE.replace('self.proj', 'self.projé')
    capture, _ = setup(tmp_path, source)
    calls = document(capture, source).semantic.metadata['sourceDeclarationCalls']
    assert len(calls['@declaration:["model.py","Block","projé"]']) == 2


def utf16(text):
    return len(text.encode('utf-16-le')) // 2


# A Greek identifier before a literal Sequential child, and an astral character
# (two UTF-16 units, four UTF-8 bytes, one code point) inside a declaration.
NON_ASCII_NET = '''from torch import nn
class Net(nn.Module):
    def __init__(self, d_μ=4):
        super().__init__()
        self.tag = nn.Identity(note="😀")
        self.seq = nn.Sequential(nn.Linear(d_μ, 4), nn.ReLU())
    def forward(self, x):
        return self.seq(self.tag(x))
'''


def test_static_spans_count_utf16_columns_after_non_ascii_text():
    # The editor places the caret by these columns. Counted in UTF-8 bytes, a
    # click on the ReLU card landed on the second `n` of `nn.ReLU()`.
    spans = project(NON_ASCII_NET, cls='Net', review_only=False)['spanmap']
    tag, seq = NON_ASCII_NET.split('\n')[4:6]
    assert (spans['tag']['start'], spans['tag']['end']) == ([5, 8], [5, utf16(tag)])
    relu = next(span for key, span in spans.items() if key.startswith('relu@'))
    assert (relu['start'], relu['end']) == ([6, utf16(seq[:seq.index('nn.ReLU')])], [6, utf16(seq[:seq.rindex(')')])])


def test_declaration_spans_and_constructor_anchors_share_one_unit():
    review = project(NON_ASCII_NET, cls='Net')
    facts = review['document']['semantic']['metadata']['reviewDeclarations']['entries']
    # A literal slot's declaration IS its constructor call: one node, one location.
    child = facts['@declaration:["model.py","Net","seq"][1]']
    assert location(child['span']) == location(child['constructor'])
    static = {location(span) for key, span in review['spanmap'].items() if not key.startswith('@')}
    assert [fact['label'] for fact in facts.values() if location(fact['span']) not in static] == []
    review_class = review['document']['semantic']['metadata']['reviewClass']
    assert review_class['members']['tag']['span']['end'] == [5, utf16(NON_ASCII_NET.split('\n')[4])]


def test_declaration_keys_spell_non_ascii_paths_as_json_stringify_does():
    # reviewDeclarations.ts remaps a host-proven Git rename by the key prefix
    # `@declaration:[${JSON.stringify(file)},`. An escaped key never matched,
    # so a renamed non-ASCII file's declarations appeared removed and added.
    from services.ide.review_declarations import review_declarations
    assert '@declaration:["données/model.py","Net","fc"]' in review_declarations({'données/model.py': NET})['entries']


def test_declarations_inside_try_with_and_match_blocks_keep_their_guards():
    # A complete inventory is authoritative in Git review, so a constructor in
    # an except fallback, a with block or a match case must be part of it.
    from services.ide.review_declarations import review_declarations
    source = '''import torch
from torch import nn
class M(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.proj = nn.Linear(d, d)
        try:
            self.attn = FlashMHA(d)
        except ImportError:
            self.attn = nn.MultiheadAttention(d, 4)
        with torch.device("meta"):
            self.head = nn.Linear(d, 2)
        match d:
            case 8:
                self.out = nn.Identity()
'''
    facts = review_declarations({'m.py': source})
    assert facts['complete']
    entries = {fact['label']: fact for fact in facts['entries'].values()}
    assert set(entries) == {'M.proj', 'M.attn', 'M.head', 'M.out'}
    # Two alternative writes stay ambiguous, as an if/else pair does.
    assert entries['M.attn']['ambiguous'] and not entries['M.head']['ambiguous']
    for before, after, label in [('except ImportError', 'except Exception', 'M.attn'),
                                 ('torch.device("meta")', 'torch.device("cpu")', 'M.head'), ('case 8', 'case 16', 'M.out')]:
        changed = {fact['label']: fact for fact in review_declarations({'m.py': source.replace(before, after)})['entries'].values()}
        assert [name for name in entries if changed[name]['digest'] != entries[name]['digest']] == [label]

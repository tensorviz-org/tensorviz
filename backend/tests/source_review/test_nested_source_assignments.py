"""Nested constructor edits remain visible without inventing runtime topology."""
import hashlib
import pytest

from tests.source_review.test_graph_review_regressions import BASE, client, project
from services.ide.projection import review_class


def source_with(statement):
    return BASE.replace('self.layer = nn.Dropout(0.1)',
                        'self.layer = nn.Dropout(0.1)\n        ' + statement)


@pytest.mark.parametrize('review_only', [True, False])
def test_nested_override_has_display_facts_in_draft_and_live_source_but_is_not_a_runtime_node(client, review_only):
    source = source_with('self.layer3[0].relu = nn.GELU()')
    source += '\nraise RuntimeError("source projection must never execute")\n'
    response = project(client, source, review_only=review_only)
    assert response.status_code == 200, response.text
    result = response.json()
    assert not result['traced']
    graph = result['document']['semantic']
    assignments = graph['metadata']['reviewClass']['assignments']
    assert set(assignments) == {'@assignment:layer3[0].relu'}
    fact = assignments['@assignment:layer3[0].relu']
    assert fact['target'] == 'layer3[0].relu'
    assert fact['title'] == 'nn.GELU'
    assert fact['value'] == 'nn.GELU()'
    assert fact['span']['start'][0] == 6
    span = result['spanmap']['@assignment:layer3[0].relu']
    assert {key: value for key, value in span.items() if key != 'focus'} == {
        **fact['span'], 'digest': fact['digest'],
        'file_digest': hashlib.sha256(source.encode()).hexdigest(),
    }
    # Code focus enriches the same assignment location without moving it.
    assert span['focus']['node'] == [[fact['span']['start'][0], fact['span']['end'][0]]]
    assert not fact['ambiguous'] and not fact['conditional']
    assert not any(node['node_id'].startswith('@assignment:') for node in graph['nodes'])
    assert not any(edge['source_node_id'].startswith('@assignment:') or edge['target_node_id'].startswith('@assignment:') for edge in graph['edges'])
    # This source assignment does not prove that a runtime layer was added.
    assert 'layer3[0].relu' not in graph['metadata']['reviewClass']['members']


@pytest.mark.parametrize('target', ['self.block.relu', 'self.blocks[0]', 'self.blocks["head"].relu'])
def test_literal_nested_targets_keep_exact_syntax_and_no_runtime_identity(target):
    facts = review_class(source_with(f'{target} = nn.GELU()'), 'Net', 'model.py')
    assignment = next(iter(facts['assignments'].values()))
    assert assignment['target'] == target[5:].replace('"', "'")
    assert not assignment['ambiguous'] and not assignment['conditional']


@pytest.mark.parametrize('statement', [
    'self.blocks[index].relu = nn.GELU()',
    'self.blocks[get_index()].relu = nn.GELU()',
    'self.blocks[-1].relu = nn.GELU()',
    'self[0].relu = nn.GELU()',
    'self[0] = nn.GELU()',
    'getattr(self, "block").relu = nn.GELU()',
    'def unrelated(self):\n            self.block.relu = nn.GELU()',
])
def test_dynamic_targets_and_nested_functions_do_not_get_a_claimed_target(statement):
    assert not review_class(source_with(statement), 'Net', 'model.py').get('assignments')


def test_guarded_and_repeated_writes_keep_uncertainty_instead_of_picking_a_winner(client):
    for statement, flag in [
        ('if enabled:\n            self.block.relu = nn.GELU()', 'conditional'),
        ('self.block.relu = nn.GELU()\n        self.block.relu = nn.Sigmoid()', 'ambiguous'),
    ]:
        response = project(client, source_with(statement), review_only=True)
        assert response.status_code == 200, response.text
        graph = response.json()['document']['semantic']
        fact = next(iter(graph['metadata']['reviewClass']['assignments'].values()))
        assert fact[flag]
        if flag == 'ambiguous':
            assert fact['title'] == 'Multiple assignments'


@pytest.mark.parametrize('later', ['self.blocks = make_blocks()', 'self.blocks[0] = make_block()', 'super().__init__()'])
def test_later_ancestor_writes_and_super_initializers_cannot_confirm_a_nested_assignment(later):
    facts = review_class(source_with(f'self.blocks[0].relu = nn.GELU()\n        {later}'), 'Net', 'model.py')
    assert facts['assignments']['@assignment:blocks[0].relu']['ambiguous']


@pytest.mark.parametrize('later', [
    'with torch.no_grad():\n            self.blocks = make_blocks()',
    'try:\n            super().__init__()\n        except RuntimeError:\n            pass',
    'match mode:\n            case 1:\n                self.blocks[0] = make_block()',
])
def test_later_writes_inside_compound_blocks_cannot_confirm_a_nested_assignment(later):
    def facts(statement):
        return review_class(source_with(statement), 'Net', 'model.py')
    assert facts(f'self.blocks[0].relu = nn.GELU()\n        {later}')['assignments']['@assignment:blocks[0].relu']['ambiguous']
    # So a capture taken before the assignment never lends it context.
    assert facts(f'self.blocks[0].relu = nn.GELU()\n        {later}')['structureDigest'] != facts(later)['structureDigest']


def test_nested_assignments_inside_compound_blocks_are_conditional_display_facts():
    facts = review_class(source_with('with torch.no_grad():\n            self.block.relu = nn.GELU()'), 'Net', 'model.py')
    fact = next(iter(facts['assignments'].values()))
    assert fact['target'] == 'block.relu' and fact['conditional'] and not fact['ambiguous']


def test_earlier_ancestor_write_and_later_sibling_do_not_make_the_target_ambiguous():
    source = source_with('self.blocks = make_blocks()\n        self.blocks[0].relu = nn.GELU()\n        self.blocks[1].relu = nn.SiLU()')
    facts = review_class(source, 'Net', 'model.py')['assignments']
    assert not any(fact['ambiguous'] for fact in facts.values())


def test_assignment_digest_changes_with_value_or_guard_but_not_formatting():
    def fact(statement):
        return next(iter(review_class(source_with(statement), 'Net', 'model.py')['assignments'].values()))
    first = fact('self.layer3[0].relu = nn.GELU()')
    assert first['digest'] == fact('self.layer3[0].relu = nn.GELU( ) # comment')['digest']
    assert first['digest'] != fact('self.layer3[0].relu = nn.Sigmoid()')['digest']
    assert first['digest'] != fact('if enabled:\n            self.layer3[0].relu = nn.GELU()')['digest']


def test_assignment_spans_count_utf16_columns_and_still_admit_capture_context():
    # A statement joined after non-ASCII text starts where VS Code counts it,
    # and the structural comparison still finds it at that location.
    def utf16(text):
        return len(text.encode('utf-16-le')) // 2
    tag = 'self.tag = nn.Identity(note="😀")'
    line = f'        {tag}; self.blocks[0].relu = nn.GELU()'
    facts = review_class(source_with(line.strip()), 'Net', 'model.py')
    fact = facts['assignments']['@assignment:blocks[0].relu']
    assert (fact['span']['start'], fact['span']['end']) == ([6, utf16(line[:line.index('self.blocks')])], [6, utf16(line)])
    assert facts['members']['tag']['span']['end'] == [6, utf16(line[:line.index(';')])]
    assert facts['structureDigest'] == review_class(source_with(tag), 'Net', 'model.py')['structureDigest']


def test_assignment_paths_are_ast_segments_not_display_labels():
    numeric = review_class(source_with('self.blocks[0].relu = nn.GELU()'), 'Net', 'model.py')
    string = review_class(source_with('self.blocks["0"].relu = nn.GELU()'), 'Net', 'model.py')
    for facts, key in [(numeric, 0), (string, '0')]:
        fact = next(iter(facts['assignments'].values()))
        assert fact['path'] == [dict(kind='attribute', key='blocks'), dict(kind='index', key=key),
                                dict(kind='attribute', key='relu')]


def test_capture_context_is_valid_only_when_literal_nested_assignments_are_the_only_change():
    def structure(statement):
        return review_class(source_with(statement), 'Net', 'model.py')['structureDigest']
    baseline = structure('')
    assert structure('self.blocks[0].relu = nn.GELU()') == baseline
    assert structure('self.blocks[0].relu = nn.SiLU()') == baseline
    for statement in [
        'self.fc2 = nn.Linear(4, 4)',
        'self.blocks[0].relu = nn.GELU(); self.fc2 = nn.Linear(4, 4)',
        'if enabled:\n            self.blocks[0].relu = nn.GELU()',
        'self.blocks[0].relu = nn.GELU()\n        self.blocks[0].relu = nn.SiLU()',
        'self.blocks[0].relu = self.other = nn.GELU()',
        'setattr(self.blocks[0], "relu", nn.GELU())',
        'self.blocks[index].relu = nn.GELU()',
    ]:
        assert structure(statement) != baseline, statement
    changed_import = source_with('').replace('from torch import nn', 'from other import nn')
    assert review_class(changed_import, 'Net', 'model.py')['structureDigest'] != baseline
    changed_forward = source_with('').replace('return', 'return 2 *', 1)
    assert review_class(changed_forward, 'Net', 'model.py')['structureDigest'] != baseline


NESTED_SOURCE = '''from torch import nn
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.relu(x + 1)
        return self.relu(x + 2)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(Block())
    def forward(self, x):
        return self.blocks(x)
'''


def capture_envelope(model, files, source_snapshot=None):
    import torch
    from services.ingest.trace import trace_model
    from services.semantic.contracts import CaptureLimits, SourceSnapshot
    from services.semantic.recorder import CaptureContext
    context = CaptureContext(source=source_snapshot or SourceSnapshot(version=1, id='source:0', workspaceSessionId='session:0',
        workspaceId='workspace:0', modelId='model:0', modelFile='model.py', className='Net', generation=0,
        files=[dict(path=file, digest=hashlib.sha256(text.encode()).hexdigest(), documentVersion=1, role='source')
               for file, text in files.items()],
        dependencyScope='complete', configDigest=None, environmentDigest='b' * 64, sourceMode='saved'),
        plan_digest='c' * 64, attempt_id='attempt:0',
        limits=CaptureLimits(operations=1000, invocations=1000, values=1000, bytes=4 * 1024 * 1024))
    traced = trace_model(model, (torch.ones(2, 4),), tier='real', semantic_context=context)
    assert traced.semantic_capture is not None, traced.semantic_error
    return dict(protocol_version=1, result=dict(graph=dict(nodes=[], edges=[]),
        diagnostics=dict(semantic_capture=traced.semantic_capture)), spans={}, error=None)


def nested_capture_projections(client, source=NESTED_SOURCE, target="self.blocks[0].relu"):
    namespace = {'__name__': 'nested_preview_test'}
    exec(compile(source, '/workspace/model.py', 'exec'), namespace)
    envelope = capture_envelope(namespace['Net'](), {'model.py': source})
    head, forward = source.rsplit('    def forward', 1)
    proposed = head + f'        {target} = nn.GELU()\n' + '    def forward' + forward
    results = {
        'capture': project(client, source, trace_result=envelope),
        'before': project(client, source, review_only=True),
        'after': project(client, proposed, review_only=True),
        'accepted': project(client, proposed),
    }
    for response in results.values():
        assert response.status_code == 200, response.text
    return {name: response.json() for name, response in results.items()}


def test_capture_and_accepted_source_share_verified_structure_and_all_reused_calls(client):
    results = nested_capture_projections(client)
    digests = {result['document']['semantic']['metadata']['reviewClass']['structureDigest'] for result in results.values()}
    assert len(digests) == 1
    capture = results['capture']['document']['semantic']
    assert capture['metadata']['registeredContainers']['blocks'] == dict(
        definition='torch.nn.modules.container.Sequential', keys=['0'])
    fact = next(f for f in capture['metadata']['sourceCorrespondence'] if f['nodeId'] == 'blocks.0.relu')
    assert fact['basis'] == 'registered_path'
    assert len(fact['invocationIds']) == 3
    assert not results['accepted']['traced']
    assert not any(n['op'] == 'GELU' for n in capture['nodes'])


@pytest.mark.parametrize('constructor', ['Block()', 'nn.Sequential(Block())', 'nn.ModuleList([Block()])', 'nn.ModuleDict({"head": Block()})'])
def test_structure_evidence_includes_consulted_dependencies(client, constructor):
    source = 'from layers import Block\n' + NESTED_SOURCE[NESTED_SOURCE.index('class Net'):]
    source = 'from torch import nn\n' + source.replace('nn.Sequential(Block())', constructor)
    dependency = NESTED_SOURCE[:NESTED_SOURCE.index('class Net')]
    before = project(client, source, {'layers.py': dependency}).json()
    after = project(client, source, {'layers.py': dependency.replace('ReLU()', 'Sigmoid()')}).json()
    def digest(result):
        return result['document']['semantic']['metadata']['reviewClass']['structureDigest']
    assert digest(before) != digest(after)


@pytest.mark.parametrize('constructor,call,target,path,keys', [
    ('nn.ModuleList([Block()])', 'self.blocks[0](x)', 'self.blocks[0].relu', 'blocks.0.relu', ['0']),
    ('nn.ModuleDict({"head": Block()})', 'self.blocks["head"](x)', 'self.blocks["head"].relu', 'blocks.head.relu', ['head']),
])
def test_real_uncalled_containers_attest_all_three_calls(client, constructor, call, target, path, keys):
    source = NESTED_SOURCE.replace('nn.Sequential(Block())', constructor).replace('self.blocks(x)', call)
    results = nested_capture_projections(client, source, target)
    graph = results['capture']['document']['semantic']
    assert graph['metadata']['registeredContainers']['blocks']['keys'] == keys
    assert not any(f['nodeId'] == 'blocks' and f['invocationIds'] for f in graph['metadata']['sourceCorrespondence'])
    fact = next(f for f in graph['metadata']['sourceCorrespondence'] if f['nodeId'] == path)
    assert fact['basis'] == 'registered_path'
    assert len(fact['invocationIds']) == 3
    assert graph['metadata']['reviewClass']['dependencyScope'] == 'complete'
    assert graph['metadata']['reviewClass']['dependencyDigests'] == {}


def test_source_evidence_includes_unexpanded_candidates_and_transitive_changes(client):
    # A helper hides the block from static lowering. Every supplied dependency
    # must still be available to compare against an invocation's attestation.
    source = NESTED_SOURCE.replace('nn.Sequential(Block())', 'build_blocks()')
    source = 'from factory import build_blocks\n' + source
    dependencies = {'factory.py': 'from layers import Block\ndef build_blocks():\n    return Block()\n',
                    'layers.py': 'from base import Base\nclass Block(Base):\n    pass\n',
                    'base.py': 'from torch import nn\nclass Base(nn.ReLU):\n    pass\n'}
    before = project(client, source, dependencies).json()['document']['semantic']['metadata']['reviewClass']
    assert before['dependencyDigests'] == {file: hashlib.sha256(text.encode()).hexdigest() for file, text in dependencies.items()}
    dependencies['base.py'] = dependencies['base.py'].replace('ReLU', 'Sigmoid')
    after = project(client, source, dependencies).json()['document']['semantic']['metadata']['reviewClass']
    assert before['dependencyDigests']['base.py'] != after['dependencyDigests']['base.py']


def multifile_capture_projections(client, monkeypatch, host_root=None):
    import sys
    import types
    base = NESTED_SOURCE[:NESTED_SOURCE.index('class Net')].replace('class Block', 'class Base')
    dependencies = {
        'review_base.py': base,
        'review_layers.py': 'from review_base import Base\nclass Block(Base):\n    pass\n',
        'review_factory.py': 'from torch import nn\nfrom review_layers import Block\ndef build_blocks():\n    return nn.Sequential(Block())\n',
    }
    source = 'from torch import nn\nfrom review_factory import build_blocks\n' + NESTED_SOURCE[NESTED_SOURCE.index('class Net'):].replace('nn.Sequential(Block())', 'build_blocks()')
    for file, text in dependencies.items():
        module = types.ModuleType(file[:-3])
        module.__file__ = '/workspace/' + file
        monkeypatch.setitem(sys.modules, file[:-3], module)
        exec(compile(text, module.__file__, 'exec'), module.__dict__)
    namespace = {'__name__': 'nested_multifile_test'}
    exec(compile(source, '/workspace/model.py', 'exec'), namespace)
    files = {'model.py': source, **dependencies}
    host_source = None
    if host_root is not None:
        from services.validation.host_snapshot import snapshot
        from services.semantic.contracts import CapturePreparation
        for file, text in files.items():
            (host_root / file).write_text(text)
        host_source = CapturePreparation.model_validate(snapshot(dict(root=str(host_root), executable=sys.executable,
            environmentRevision=0, session='session', workspaceId='workspace', runId='attempt', generation=1,
            modelFile='model.py', className='Net', timeoutMs=30000,
            captureRequest=dict(input=None, tensorInputs=None, overrides=None)))['preparation']).source
    envelope = capture_envelope(namespace['Net'](), files, host_source)
    proposed = source.replace('self.blocks = build_blocks()', 'self.blocks = build_blocks()\n        self.blocks[0].relu = nn.GELU()')
    captured = project(client, source, dependencies, trace_result=envelope)
    current = project(client, proposed, dependencies, review_only=True)
    assert captured.status_code == current.status_code == 200, (captured.text, current.text)
    return captured.json(), current.json(), envelope, source, dependencies


def test_real_multifile_capture_attests_transitive_base_and_helper_dependencies(client, monkeypatch):
    captured, current, envelope, source, dependencies = multifile_capture_projections(client, monkeypatch)
    def facts(result):
        return result['document']['semantic']['metadata']['reviewClass']
    before, after = facts(captured), facts(current)
    assert before['dependencyScope'] == 'complete'
    assert before['structureDigest'] == after['structureDigest']
    assert before['dependencyDigests'] == after['dependencyDigests']
    assert set(before['dependencyDigests']) == set(dependencies)
    correspondence = captured['document']['semantic']['metadata']['sourceCorrespondence']
    assert len(next(f for f in correspondence if f['nodeId'] == 'blocks.0.relu')['invocationIds']) == 3
    # The imported base was reached at runtime through a helper and container.
    # A changed base invalidates its attestation even if static lowering never expanded it.
    dependencies['review_base.py'] = dependencies['review_base.py'].replace('ReLU()', 'Sigmoid()')
    changed = project(client, source, dependencies, review_only=True).json()
    assert facts(changed)['dependencyDigests']['review_base.py'] != before['dependencyDigests']['review_base.py']
    # An incomplete candidate set must not silently drop a captured dependency.
    missing = {file: text for file, text in dependencies.items() if file != 'review_base.py'}
    partial = project(client, source, missing, trace_result=envelope)
    assert partial.status_code == 200, partial.text
    assert facts(partial.json())['dependencyDigests'] == before['dependencyDigests']
    # Its source-only counterpart is incomplete, so the renderer will refuse context.
    assert 'review_base.py' not in facts(project(client, source, missing, review_only=True).json())['dependencyDigests']


def host_capture_projections(client, tmp_path):
    import sys
    from services.validation.host_snapshot import snapshot
    from services.semantic.contracts import CapturePreparation
    (tmp_path / 'model.py').write_text(NESTED_SOURCE)
    # Production preflight snapshots a broader set than the projection's 256-file
    # candidate budget. They must not be mistaken for complete runtime imports.
    for i in range(260):
        (tmp_path / f'unrelated_{i}.py').write_text('unused = True\n')
    preparation = CapturePreparation.model_validate(snapshot(dict(root=str(tmp_path), executable=sys.executable,
        environmentRevision=0, session='session', workspaceId='workspace', runId='attempt', generation=1,
        modelFile='model.py', className='Net', timeoutMs=30000,
        captureRequest=dict(input=None, tensorInputs=None, overrides=None)))['preparation'])
    assert preparation.source.dependencyScope == 'partial'
    assert len(preparation.source.files) == 261
    namespace = {'__name__': 'host_preview_test'}
    exec(compile(NESTED_SOURCE, str(tmp_path / 'model.py'), 'exec'), namespace)
    envelope = capture_envelope(namespace['Net'](), {'model.py': NESTED_SOURCE}, preparation.source)
    captured = project(client, NESTED_SOURCE, trace_result=envelope)
    proposed = NESTED_SOURCE.replace('self.blocks = nn.Sequential(Block())',
        'self.blocks = nn.Sequential(Block())\n        self.blocks[0].relu = nn.GELU()')
    after = project(client, proposed, review_only=True)
    assert captured.status_code == after.status_code == 200, (captured.text, after.text)
    return captured.json(), after.json()


def test_production_host_snapshot_partial_scope_and_candidate_budget_are_preserved(client, tmp_path):
    captured, after = host_capture_projections(client, tmp_path)
    facts = captured['document']['semantic']['metadata']['reviewClass']
    assert facts['dependencyScope'] == 'partial'
    assert facts['dependencyDigests'] == {}  # The supplied, verified candidates, not all workspace files.
    assert facts['structureDigest'] == after['document']['semantic']['metadata']['reviewClass']['structureDigest']


def test_unregistered_reference_can_keep_calling_the_previous_module_after_override(client):
    source = NESTED_SOURCE.replace('self.relu = nn.ReLU()', 'self.relu = nn.ReLU()\n        self.acts = [self.relu]')
    source = source.replace('self.relu(', 'self.acts[0](')
    proposed = source.replace('self.blocks = nn.Sequential(Block())',
        'self.blocks = nn.Sequential(Block())\n        self.blocks[0].relu = nn.GELU()')
    results = nested_capture_projections(client, proposed)
    # A plain Python list keeps the old module alive: assigning GELU is not proof
    # that any GELU call happens. UI coverage must call this a source assignment.
    nodes = results['capture']['document']['semantic']['nodes']
    assert any('relu' in str(node['op']).lower() for node in nodes)
    assert not any('gelu' in str(node['op']).lower() for node in nodes)


def test_real_partial_multifile_capture_tracks_changed_and_missing_candidates(client, monkeypatch, tmp_path):
    captured, _, envelope, source, dependencies = multifile_capture_projections(client, monkeypatch, tmp_path)
    facts = captured['document']['semantic']['metadata']['reviewClass']
    assert facts['dependencyScope'] == 'partial'
    assert facts['dependencyDigests'] == {file: hashlib.sha256(text.encode()).hexdigest() for file, text in dependencies.items()}
    changed = {**dependencies, 'review_base.py': dependencies['review_base.py'].replace('ReLU()', 'Sigmoid()')}
    # CaptureJoin refuses candidates that claim to belong to this run but changed.
    assert project(client, source, changed, trace_result=envelope).status_code == 400
    current = project(client, source, changed, review_only=True).json()['document']['semantic']['metadata']['reviewClass']
    assert current['dependencyDigests']['review_base.py'] != facts['dependencyDigests']['review_base.py']
    missing = {file: text for file, text in dependencies.items() if file != 'review_base.py'}
    current = project(client, source, missing, review_only=True).json()['document']['semantic']['metadata']['reviewClass']
    assert 'review_base.py' not in current['dependencyDigests']

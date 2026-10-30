"""Exact multi-file candidates reuse the normal worker without recipe claims."""
import hashlib
import sys

import pytest

from services.semantic.candidate import stage_candidate, CandidateRefused
from services.semantic.contracts import SourceCandidateRequest, SourceCandidateExecution, CaptureSnapshot
from tests.source_review.test_semantic_candidate import prepare
from tests.source_review.test_semantic_candidate_worker import replay

CONFIG_FILE = '_tensorviz_candidate_width.py'
CONFIG = 'WIDTH = 12\nHEADS = 3\n'
SOURCE = '''import torch
from _tensorviz_candidate_width import WIDTH, HEADS
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(4, WIDTH)
        self.down = torch.nn.Linear(12, 4)
    def forward(self, x):
        x = self.up(x).reshape(x.shape[0], HEADS, -1)
        return self.down(x.flatten(1))
'''


@pytest.fixture
def base(tmp_path, monkeypatch):
    root = (tmp_path / 'original').resolve()
    root.mkdir()
    (root / CONFIG_FILE).write_text(CONFIG)
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop('_tensorviz_candidate_width', None)
    try:
        plan, capture = prepare(root, SOURCE)
        yield root, plan, capture
    finally:
        sys.modules.pop('_tensorviz_candidate_width', None)


def request(plan, capture, replacements=None):
    replacements = replacements or {
        'model.py': SOURCE.replace('Linear(12, 4)', 'Linear(16, 4)'),
        CONFIG_FILE: 'WIDTH = 16\nHEADS = 4\n',
    }
    return SourceCandidateRequest(version=2, id='candidate:source-set', plan=plan, capture=capture,
        reviewedSources=[dict(file=file, beforeDigest=next(item.digest for item in plan.source.files if item.path == file),
            source=source) for file, source in replacements.items()])


def test_staging_is_static_and_pins_every_reviewed_byte(base, tmp_path):
    root, plan, capture = base
    marker = tmp_path / 'must-not-execute'
    value = request(plan, capture)
    value.reviewedSources[0].source += f'\nopen({str(marker)!r}, "w").write("executed")\n'
    with stage_candidate(str(root), plan, capture, reviewed_sources=value.reviewedSources,
            candidate_id=value.id) as staged:
        execution = SourceCandidateExecution.model_validate(staged.execution.model_dump())
        assert staged.postconditions is None
        assert not marker.exists()
        assert [item.file for item in execution.changes] == sorted(item.file for item in value.reviewedSources)
        for item in value.reviewedSources:
            assert (staged.workspace / item.file).read_bytes() == item.source.encode()
            identity = next(change for change in execution.changes if change.file == item.file)
            assert identity.beforeDigest == item.beforeDigest
            assert identity.afterDigest == hashlib.sha256(item.source.encode()).hexdigest()
        assert execution.plan.model_dump(exclude={'id', 'source'}) == plan.model_dump(exclude={'id', 'source'})
        assert execution.plan.source.id != plan.source.id
    assert not staged.workspace.parent.exists()
    assert (root / 'model.py').read_text() == SOURCE
    assert (root / CONFIG_FILE).read_text() == CONFIG


def test_two_file_width_change_runs_exact_candidate_without_recipe_verdict(base, tmp_path):
    root, plan, capture = base
    value = request(plan, capture)
    process, envelope = replay(root, value.model_dump(mode='json'), tmp_path)
    assert process.returncode == 0, process.stderr
    execution = SourceCandidateExecution.model_validate(envelope['worker']['candidateExecution'])
    actual = CaptureSnapshot.model_validate(envelope['result']['diagnostics']['semantic_capture'])
    assert actual.status == 'complete'
    assert actual.source == execution.plan.source and actual.attemptId == value.id
    assert envelope['worker']['candidateBaseCurrent'] is True
    assert envelope['result']['diagnostics']['execution']['planCurrent'] is True
    assert 'candidatePostconditions' not in envelope['worker']
    assert len(execution.changes) == 2
    assert (root / 'model.py').read_text() == SOURCE
    assert (root / CONFIG_FILE).read_text() == CONFIG


@pytest.mark.parametrize('change', ['head_divisibility', 'downstream_width', 'missing_model_class'])
def test_inconsistent_rewrites_preserve_runtime_failure_and_never_claim_verification(base, tmp_path, change):
    root, plan, capture = base
    replacements = {'model.py': SOURCE.replace('Linear(12, 4)', 'Linear(16, 4)'),
        CONFIG_FILE: 'WIDTH = 16\nHEADS = 4\n'}
    if change == 'head_divisibility': replacements[CONFIG_FILE] = 'WIDTH = 16\nHEADS = 3\n'
    if change == 'downstream_width': replacements = {CONFIG_FILE: replacements[CONFIG_FILE]}
    if change == 'missing_model_class': replacements['model.py'] = SOURCE.replace('class Model', 'class Other')
    process, envelope = replay(root, request(plan, capture, replacements).model_dump(mode='json'), tmp_path)
    assert process.returncode != 0
    execution = SourceCandidateExecution.model_validate(envelope['worker']['candidateExecution'])
    assert execution.baseCaptureId == capture.id
    assert envelope['error'] is not None
    assert 'candidatePostconditions' not in envelope['worker']
    if change != 'missing_model_class':
        actual = CaptureSnapshot.model_validate(envelope['result']['diagnostics']['semantic_capture'])
        assert actual.status == 'partial'
        # Reshape errors currently use the generic execution diagnostic; the
        # linear dimension rule has a structured invalid-shape diagnostic.
        expected = 'execution_failed' if change == 'head_divisibility' else 'invalid_shape'
        assert any(item.code == expected for item in actual.diagnostics)
    assert (root / 'model.py').read_text() == SOURCE


@pytest.mark.parametrize('change', ['empty', 'duplicate', 'untracked', 'before_digest', 'unchanged',
    'syntax', 'non_python', 'per_file_bytes', 'unicode_bytes', 'total_bytes', 'too_many', 'extra_field', 'competing'])
def test_source_set_refusals_are_static_bounded_and_leave_no_candidate(base, tmp_path, change):
    root, plan, capture = base
    items = [item.model_dump() for item in request(plan, capture).reviewedSources]
    marker = tmp_path / 'private-must-not-run'
    items[0]['source'] += f'\nopen({str(marker)!r}, "w").write("ran")\n'
    options = {}
    if change == 'empty': items = []
    if change == 'duplicate': items.append(items[0])
    if change == 'untracked': items[0]['file'] = 'untracked.py'
    if change == 'before_digest': items[0]['beforeDigest'] = '0' * 64
    if change == 'unchanged': items[0]['source'] = SOURCE
    if change == 'syntax': items[0]['source'] = 'return private_source_text()'
    if change == 'non_python': items[0]['file'] = 'tensorviz.toml'
    if change == 'per_file_bytes': items[0]['source'] = '#' * 262145
    if change == 'unicode_bytes': items[0]['source'] = '#' + '\U0001f642' * 65536
    if change == 'total_bytes': items = [dict(items[0], file=f'file{i}.py', source='#' * 262144) for i in range(5)]
    if change == 'too_many': items = [dict(items[0], file=f'file{i}.py') for i in range(17)]
    if change == 'extra_field': items[0]['afterDigest'] = '0' * 64
    if change == 'competing': options['intent'] = {'private': 'unexpected'}
    if change in {'empty', 'duplicate', 'non_python', 'per_file_bytes', 'unicode_bytes', 'total_bytes', 'too_many', 'extra_field'}:
        # Establish the bound itself, independently of untracked-file rejection.
        with pytest.raises(ValueError):
            SourceCandidateRequest(version=2, id='candidate:limits', plan=plan, capture=capture, reviewedSources=items)
    with pytest.raises(ValueError):
        with stage_candidate(str(root), plan, capture, reviewed_sources=items, temporary_parent=str(tmp_path), **options):
            pytest.fail('invalid candidate was staged')
    assert not marker.exists()
    assert not list(tmp_path.glob('tensorviz-candidate-*'))
    assert (root / 'model.py').read_text() == SOURCE
    assert (root / CONFIG_FILE).read_text() == CONFIG


@pytest.mark.parametrize('file', ['model.py', CONFIG_FILE])
def test_either_stale_reviewed_file_refuses_before_execution(base, tmp_path, file):
    root, plan, capture = base
    value = request(plan, capture)
    with (root / file).open('a') as stream:
        stream.write('\n# changed after review\n')
    with pytest.raises(CandidateRefused, match='candidate_base_changed'):
        with stage_candidate(str(root), plan, capture, reviewed_sources=value.reviewedSources,
                temporary_parent=str(tmp_path)):
            pytest.fail('stale base was staged')
    assert not list(tmp_path.glob('tensorviz-candidate-*'))


@pytest.mark.parametrize('change', ['duplicate', 'order', 'missing_source', 'digest', 'unchanged', 'attempt', 'source'])
def test_execution_contract_rejects_inconsistent_manifest(base, change):
    root, plan, capture = base
    with stage_candidate(str(root), plan, capture, reviewed_sources=request(plan, capture).reviewedSources) as staged:
        data = staged.execution.model_dump(mode='json')
    if change == 'duplicate': data['changes'].append(data['changes'][0])
    if change == 'order': data['changes'].reverse()
    if change == 'missing_source': data['changes'][0]['file'] = 'missing.py'
    if change == 'digest': data['changes'][0]['afterDigest'] = '0' * 64
    if change == 'unchanged': data['changes'][0]['beforeDigest'] = data['changes'][0]['afterDigest']
    if change == 'attempt': data['id'] = data['plan']['id'] = data['baseAttemptId']
    if change == 'source': data['plan']['source']['id'] = data['baseSourceId']
    with pytest.raises(ValueError): SourceCandidateExecution.model_validate(data)


@pytest.mark.parametrize('which', ['original', 'candidate'])
def test_runtime_changes_to_either_workspace_cannot_claim_current_sources(base, tmp_path, which):
    root, plan, capture = base
    value = request(plan, capture)
    destination = repr(str(root / CONFIG_FILE)) if which == 'original' else f'str(Path(__file__).parent / {CONFIG_FILE!r})'
    value.reviewedSources[0].source = value.reviewedSources[0].source.replace('        x = self.up(x)',
        '        from pathlib import Path\n'
        f'        Path({destination}).write_text("# changed during candidate forward")\n'
        '        x = self.up(x)')
    process, envelope = replay(root, value.model_dump(mode='json'), tmp_path)
    assert process.returncode == 0, process.stderr
    assert envelope['worker']['candidateBaseCurrent'] is (which != 'original')
    assert envelope['result']['diagnostics']['execution']['planCurrent'] is (which != 'candidate')
    assert 'candidatePostconditions' not in envelope['worker']

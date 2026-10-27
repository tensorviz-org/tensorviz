"""Initializer entry records copied scalar bindings without evaluating objects."""
import gc
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import weakref

import pytest
from pydantic import ValidationError

from services.semantic.contracts import CaptureSnapshot, InitializerArguments, PortableCapture
from services.semantic.portable import portable_capture
from tests.capture.test_semantic_construction import construct
from tests.source_review.test_semantic_source_bindings import run

SOURCE = '''import torch
class Block(torch.nn.Module):
    def __init__(this, width=4, /, *sizes, bias=False, **options):
        super().__init__()
        width = 999
        options["width"] = 999
    def forward(self, h):
        return h
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Block(4, 8, width=3, bias=True, probability=0.25)
    def forward(self, h):
        return self.a(h)
def create():
    return Model()
'''


def test_entry_preserves_positional_only_defaults_variadics_and_non_self_receiver(tmp_path):
    model, context, _ = construct(tmp_path, SOURCE)
    capture = run(model, context)
    assert capture.status == 'complete'
    arguments = next(module for module in capture.modules if 'a' in module.paths).initializerArguments
    assert arguments.phase == 'initializer_entry' and arguments.status == 'observed'
    assert arguments.positionalOnly == 1
    assert [(item.name, item.value.value) for item in arguments.positional] == [('width', '4')]
    assert [(item.name, item.value.value) for item in arguments.keywordOnly] == [('bias', True)]
    assert [item.value for item in arguments.varArgs] == ['8']
    assert {name: value.value for name, value in arguments.varKwargs.items()} == {'width': '3', 'probability': 0.25}
    # These values were changed inside __init__; snapshots must describe entry.
    model, context, _ = construct(tmp_path, SOURCE.replace('Block(4, 8, width=3, bias=True, probability=0.25)', 'Block()'))
    defaults = next(module for module in run(model, context).modules if 'a' in module.paths).initializerArguments
    assert defaults.positional[0].value.value == '4' and defaults.keywordOnly[0].value.value is False
    assert defaults.varArgs == [] and defaults.varKwargs == {}


def test_unsupported_objects_strings_and_tensors_are_opaque_without_callbacks_or_retention(tmp_path):
    source = SOURCE.replace('class Block', '''class Hostile:
    @property
    def __class__(self): raise RuntimeError("descriptor executed")
    def __repr__(self): raise RuntimeError("repr executed")
    def __str__(self): raise RuntimeError("str executed")
class Block''').replace('Block(4, 8, width=3, bias=True, probability=0.25)',
    'Block(Hostile(), torch.ones(2), text="private-text", options=[1, 2])')
    model, context, _ = construct(tmp_path, source)
    snapshot = context.source_evidence.construction.arguments(model.a)
    assert snapshot['positional'][0]['value'] == {'kind': 'opaque', 'reason': 'unsupported_type', 'typeName': None}
    assert snapshot['varArgs'][0]['kind'] == 'opaque'
    assert all(value['kind'] == 'opaque' for value in snapshot['varKwargs'].values())
    assert 'private-text' not in str(snapshot)
    reference = weakref.ref(model.a)
    del model
    gc.collect()
    assert reference() is None and not context.source_evidence.construction.instances


@pytest.mark.parametrize('number,kind,value', [('12345678901234567890', 'int', '12345678901234567890'),
    ('-0.0', 'float', '-0'), ('float("inf")', 'float', 'inf'), ('True', 'bool', True), ('None', 'none', None)])
def test_scalar_grammar_preserves_types_precision_and_signed_zero(tmp_path, number, kind, value):
    model, context, _ = construct(tmp_path, SOURCE.replace('Block(4, 8, width=3, bias=True, probability=0.25)', 'Block(' + number + ')'))
    item = next(module for module in run(model, context).modules if 'a' in module.paths).initializerArguments.positional[0].value
    assert item.kind == kind and getattr(item, 'value', None) == value


@pytest.mark.parametrize('budget', ['MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES'])
def test_argument_byte_bounds_keep_forward_and_creation_provenance_available(tmp_path, monkeypatch, budget):
    monkeypatch.setattr('services.semantic.initializer_arguments.' + budget, 1)
    model, context, _ = construct(tmp_path, SOURCE)
    capture = run(model, context)
    assert capture.status == 'complete'
    assert all(module.constructorClassVerified for module in capture.modules)
    assert all(module.initializerArguments.status == 'unavailable' and module.initializerArguments.reason == 'byte_limit'
        for module in capture.modules)


def test_variadic_item_limit_is_not_a_truncated_argument_claim(tmp_path):
    model, context, _ = construct(tmp_path, SOURCE.replace('Block(4, 8, width=3, bias=True, probability=0.25)', 'Block(4, *range(257))'))
    capture = run(model, context)
    arguments = next(module for module in capture.modules if 'a' in module.paths).initializerArguments
    assert arguments.status == 'unavailable' and arguments.reason == 'item_limit' and arguments.varArgs is None
    assert capture.status == 'complete'


def test_non_utf8_variadic_key_does_not_break_capture_serialization(tmp_path):
    model, context, _ = construct(tmp_path, SOURCE.replace('Block(4, 8, width=3, bias=True, probability=0.25)',
        'Block(4, **{chr(0xD800): 1})'))
    capture = run(model, context)
    entry = next(module for module in capture.modules if 'a' in module.paths).initializerArguments
    assert entry.status == 'unavailable' and entry.reason == 'unsupported_type'


def test_bound_parameter_work_limits_do_not_invoke_key_protocols():
    from types import SimpleNamespace
    from services.semantic.initializer_arguments import observe_parameters
    class Key(str):
        def __eq__(self, other): raise RuntimeError('key equality executed')
        __hash__ = str.__hash__
    def initializer(self, **options): pass
    # Isolate observation from Python's own keyword binder, which may compare
    # subclassed string keys before the profile entry event.
    frame = SimpleNamespace(f_code=initializer.__code__, f_locals={'self': None, 'options': {Key('extra'): 1}})
    entry, _ = observe_parameters(frame, 16384)
    assert entry['status'] == 'unavailable' and entry['reason'] == 'unsupported_type'


def test_reinitialization_or_class_changes_invalidate_argument_evidence(tmp_path):
    model, context, namespace = construct(tmp_path, SOURCE)
    observer = context.source_evidence.construction
    assert observer.arguments(model.a) is not None
    with observer.observe():
        namespace['Block'].__init__(model.a)
    assert observer.arguments(model.a) is None
    model, context, namespace = construct(tmp_path, SOURCE)
    class Other(namespace['Block']): pass
    model.a.__class__ = Other
    assert context.source_evidence.construction.arguments(model.a) is None


@pytest.mark.parametrize('change', ['names', 'positional_only', 'keyword_collision', 'status', 'reason', 'object_metadata', 'unbound'])
def test_contract_rejects_unsupported_or_contradictory_initializer_claims(tmp_path, change):
    model, context, _ = construct(tmp_path, SOURCE)
    payload = run(model, context).model_dump(mode='json')
    module = next(item for item in payload['modules'] if 'a' in item['paths'])
    arguments = module['initializerArguments']
    if change == 'names': arguments['keywordOnly'][0]['name'] = 'width'
    if change == 'positional_only': arguments['positionalOnly'] = 2
    if change == 'keyword_collision': arguments['varKwargs']['bias'] = {'kind': 'bool', 'value': True}
    if change == 'status': arguments['status'] = 'unavailable'
    if change == 'reason': arguments['reason'] = 'byte_limit'
    if change == 'object_metadata': arguments['positional'][0]['value'] = {'kind': 'opaque', 'reason': 'unsupported_type', 'typeName': 'private-name'}
    if change == 'unbound': module['constructorClassVerified'] = False
    with pytest.raises(ValidationError): CaptureSnapshot.model_validate(payload)


def test_legacy_captures_do_not_gain_initializer_argument_evidence(tmp_path):
    model, context, _ = construct(tmp_path, SOURCE)
    payload = run(model, context).model_dump(mode='json')
    for item in payload['modules']: item.pop('initializerArguments')
    assert all(item.initializerArguments is None for item in CaptureSnapshot.model_validate(payload).modules)


def test_unicode_signature_is_observed_at_entry_without_changing_forward(tmp_path):
    model, context, _ = construct(tmp_path, SOURCE.replace('width', 'π').replace('bias', '𝒜'))
    entry = next(module for module in run(model, context).modules if 'a' in module.paths).initializerArguments
    # Python normalizes identifiers in executable signatures (𝒜 becomes A).
    assert entry.positional[0].name == 'π' and entry.keywordOnly[0].name == 'A'
    assert entry.varKwargs['π'].value == '3'


def test_metadata_does_not_take_the_forward_capture_byte_budget(tmp_path, monkeypatch):
    from services.semantic.recorder import SemanticRecorder
    accounted = []
    original = SemanticRecorder.snapshot
    def snapshot(recorder):
        result = original(recorder)
        accounted.append(recorder.bytes)
        return result
    monkeypatch.setattr(SemanticRecorder, 'snapshot', snapshot)
    source = SOURCE.replace('Block(4, 8, width=3, bias=True, probability=0.25)', 'Block(4, *range(256))')
    source = source.replace('return self.a(h)', 'h = self.a(h)\n        for _ in range(80): h = h + 1\n        return h')
    model, context, _ = construct(tmp_path, source)
    observer = context.source_evidence.construction
    with monkeypatch.context() as patch:
        patch.setattr(observer, 'arguments', lambda module: None)
        baseline = run(model, context)
        context = replace(context, limits=context.limits.model_copy(update={'bytes': accounted[-1] + 128}))
        baseline = run(model, context)
    capture = run(model, context)
    assert capture.status == 'complete' and len(capture.operations) == len(baseline.operations) == 80
    assert len(capture.model_dump_json().encode()) <= context.limits.bytes


def test_metadata_has_an_independent_cumulative_budget(tmp_path):
    calls = ', '.join(['Block(4, *range(100))'] * 8)
    source = SOURCE.replace('self.a = Block(4, 8, width=3, bias=True, probability=0.25)',
        f'self.blocks = torch.nn.ModuleList([{calls}])')
    source = source.replace('return self.a(h)', 'for block in self.blocks: h = block(h)\n        return h')
    model, context, _ = construct(tmp_path, source)
    baseline = run(model, context)
    requested = [module.initializerArguments for module in baseline.modules if module.initializerArguments is not None]
    assert requested and all(entry.status == 'observed' for entry in requested)
    assert len([module for module in baseline.modules if any(path.startswith('blocks.') for path in module.paths)
                and module.initializerArguments is not None]) == 8
    demand = sum(len(entry.model_dump_json().encode()) for entry in requested)
    # Keep global capture ample while deliberately exceeding the independent
    # metadata allowance. Explicit calls ensure all eight constructors are evidenced;
    # a comprehension can omit them and never exercise this limit.
    context = replace(context, limits=context.limits.model_copy(update={'bytes': 8 * (demand // 2)}))
    assert demand > context.limits.bytes // 8
    capture = run(model, context)
    assert capture.status == 'complete'
    entries = [module.initializerArguments for module in capture.modules if module.initializerArguments is not None]
    assert any(entry.status == 'observed' for entry in entries)
    assert any(entry.status == 'unavailable' for entry in entries)
    assert sum(len(entry.model_dump_json().encode()) for entry in entries) <= context.limits.bytes // 8


def test_portable_roundtrip_preserves_entry_evidence_and_does_not_retain_opaque_payloads(tmp_path):
    source = SOURCE.replace('probability=0.25', 'probability=0.25, password="secret", path="/Users/private"')
    model, context, _ = construct(tmp_path, source)
    capture = run(model, context)
    portable = portable_capture(capture)
    assert portable.capture.modules == capture.modules and portable.capture.status == 'complete'
    assert 'secret' not in portable.model_dump_json() and '/Users/private' not in portable.model_dump_json()
    assert PortableCapture.model_validate_json(portable.model_dump_json()) == portable


def test_embedding_private_default_does_not_break_portable_export(tmp_path):
    source = SOURCE.replace('self.a = Block(4, 8, width=3, bias=True, probability=0.25)', 'self.a = torch.nn.Embedding(4, 3)')
    source = source.replace('return self.a(h)', 'return self.a(h.long())')
    model, context, _ = construct(tmp_path, source)
    capture = run(model, context)
    entry = next(module for module in capture.modules if 'a' in module.paths).initializerArguments
    assert any(parameter.name == '_freeze' for parameter in entry.positional + entry.keywordOnly)
    assert portable_capture(capture).capture.modules == capture.modules


FIXTURE = json.loads((Path(__file__).parent.resolve().parents[2] / 'tests/fixtures/contracts/initializer-arguments.json').read_text())


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda case: case['name'])
def test_host_and_browser_initializer_cases_agree_with_python(case):
    payload = json.loads((Path(__file__).parent.resolve().parents[2] / 'tests/fixtures/contracts/portable-capture.json').read_text())
    capture = payload['capture']
    entry = deepcopy(FIXTURE['entry'])
    for patch in case['patches']:
        target = entry
        for key in patch['path'][:-1]: target = target[key]
        target[patch['path'][-1]] = patch['value']
    binding = dict(id='constructor:0', snapshotId=capture['source']['id'], kind='constructor', quality='exact',
        anchors=[dict(file=capture['source']['files'][0]['path'], digest=capture['source']['files'][0]['digest'],
            startLine=1, startColumn=0, endLine=1, endColumn=5, scope='Model.__init__', tokenDigest='a' * 64)])
    capture['sourceBindings'].append(binding)
    capture['modules'][0].update(constructorClassVerified=True, constructorBindingIds=[binding['id']], initializerArguments=entry)
    for contract, valid in [(CaptureSnapshot, case['live']), (PortableCapture, case['portable'])]:
        value = capture if contract is CaptureSnapshot else payload
        if valid: contract.model_validate(value)
        else:
            with pytest.raises(ValidationError): contract.model_validate(value)


def test_numeric_negative_zero_cannot_lose_its_sign_at_the_wire_boundary():
    entry = deepcopy(FIXTURE['entry'])
    entry['varArgs'][0]['value'] = -0.0
    with pytest.raises(ValidationError, match='explicit string tag'):
        InitializerArguments.model_validate(entry)

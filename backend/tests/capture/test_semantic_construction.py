"""Actual module creation locations are optional provenance, never inferred paths."""
import gc
import sys
import weakref

import pytest
import torch
from pydantic import ValidationError

from services.semantic.contracts import CaptureSnapshot
from services.semantic.portable import portable_capture
from tests.source_review.test_semantic_source_bindings import prepared, run


SOURCE = '''import torch
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.ReLU()
    def forward(self, h):
        return self.activation(h)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Block()
        self.b = Block()
        self.shared = self.a
    def forward(self, h):
        return self.b(self.shared(self.a(h)))
def create():
    return Model()
'''


def construct(tmp_path, source=SOURCE):
    namespace, context = prepared(tmp_path, source)
    with context.source_evidence.construction.observe():
        model = namespace['create']()
    return model, context, namespace


def test_actual_instances_keep_constructor_sites_and_aliases_share_one_site(tmp_path):
    model, context, _ = construct(tmp_path)
    capture = run(model, context)
    bindings = {b.id: b for b in capture.sourceBindings}
    sites = {path: bindings[m.constructorBindingIds[0]].anchors[0] for m in capture.modules for path in m.paths}
    assert sites[''].startLine == 17
    assert sites['a'].startLine == sites['shared'].startLine == 11
    assert sites['b'].startLine == 12
    assert sites['a.activation'].startLine == sites['b.activation'].startLine == 5
    assert sites['a.activation'].scope == 'Block.__init__'
    assert all(m.constructorClassVerified for m in capture.modules)
    assert all(b.kind == 'constructor' for b in bindings.values() if b.id.startswith('constructor:'))
    assert len([m for m in capture.modules if 'a' in m.paths]) == 1
    # Initializer/super calls never appear as tensor operations or invocations.
    assert len(capture.operations) == 3 and all(op.target == 'aten.relu.default' for op in capture.operations)
    portable = portable_capture(capture).capture
    assert [m.constructorBindingIds for m in portable.modules] == [m.constructorBindingIds for m in capture.modules]
    assert all(m.constructorClassVerified for m in portable.modules)
    assert all(a.file == 'file-0.py' for b in portable.sourceBindings for a in b.anchors)


def test_unobserved_or_unverified_construction_is_not_recovered_from_attribute_names(tmp_path):
    namespace, context = prepared(tmp_path, SOURCE)
    model = namespace['create']()
    capture = run(model, context)
    assert not any(m.constructorBindingIds for m in capture.modules)
    assert not any(m.constructorClassVerified for m in capture.modules)
    # Test caller is outside the supplied snapshot, even though children have
    # verified constructors inside Model.__init__.
    namespace, context = prepared(tmp_path, SOURCE)
    with context.source_evidence.construction.observe():
        model = namespace['Model']()
    capture = run(model, context)
    assert not next(m for m in capture.modules if '' in m.paths).constructorBindingIds
    assert next(m for m in capture.modules if 'a' in m.paths).constructorBindingIds


def test_reinitializing_an_object_drops_ambiguous_creation_evidence(tmp_path):
    model, context, namespace = construct(tmp_path)
    with context.source_evidence.construction.observe():
        namespace['Block'].__init__(model.a)
    capture = run(model, context)
    assert not next(m for m in capture.modules if 'a' in m.paths).constructorBindingIds
    assert not next(m for m in capture.modules if 'a' in m.paths).constructorClassVerified
    assert next(m for m in capture.modules if 'b' in m.paths).constructorBindingIds


def test_observer_restores_profile_on_success_and_failure_and_never_displaces_an_existing_profiler(tmp_path):
    namespace, context = prepared(tmp_path, SOURCE)
    observer = context.source_evidence.construction
    assert sys.getprofile() is None
    with pytest.raises(RuntimeError, match='construction failed'):
        with observer.observe():
            namespace['create']()
            raise RuntimeError('construction failed')
    assert sys.getprofile() is None
    events = []
    def previous(frame, event, result):
        if event == 'call' and frame.f_code.co_name == 'create':
            events.append(event)
    sys.setprofile(previous)
    try:
        with observer.observe():
            model = namespace['create']()
        assert sys.getprofile() is previous and events
        assert observer.binding(model) is None
        assert observer.arguments(model) is None
    finally:
        sys.setprofile(None)


def test_replaced_observer_does_not_restore_over_the_user_hook_or_keep_incomplete_evidence(tmp_path):
    namespace, context = prepared(tmp_path, SOURCE)
    observer = context.source_evidence.construction
    try:
        with observer.observe():
            model = namespace['create']()
            sys.setprofile(None)
        assert sys.getprofile() is None
        assert not observer.instances and observer.binding(model) is None
        assert observer.arguments(model) is None
    finally:
        sys.setprofile(None)


def test_module_objects_and_tensor_payloads_are_not_kept_alive(tmp_path):
    model, context, _ = construct(tmp_path)
    reference = weakref.ref(model)
    observer = context.source_evidence.construction
    assert observer.instances
    del model
    gc.collect()
    assert reference() is None and not observer.instances


def test_observation_never_evaluates_an_unrelated_class_descriptor(tmp_path):
    namespace, context = prepared(tmp_path, SOURCE)
    class Unrelated:
        def __init__(self):
            pass
        @property
        def __class__(self):
            pytest.fail('construction observation evaluated a user descriptor')
    with context.source_evidence.construction.observe():
        Unrelated()
        model = namespace['create']()
    assert context.source_evidence.construction.binding(model) is not None


def test_unavailable_profile_hook_keeps_execution_available(tmp_path, monkeypatch):
    namespace, context = prepared(tmp_path, SOURCE)
    def denied(_hook):
        raise RuntimeError('profile installation denied')
    monkeypatch.setattr(sys, 'setprofile', denied)
    with context.source_evidence.construction.observe():
        model = namespace['create']()
    assert not context.source_evidence.construction.active
    assert context.source_evidence.construction.binding(model) is None
    assert context.source_evidence.construction.arguments(model) is None
    assert run(model, context).status == 'complete'


def test_constructor_site_budget_exhaustion_does_not_prevent_forward_or_guess_old_sites(tmp_path, monkeypatch):
    monkeypatch.setattr('services.semantic.construction.MAX_CONSTRUCTIONS', 1)
    model, context, _ = construct(tmp_path)
    capture = run(model, context)
    assert capture.status == 'complete' and len(capture.operations) == 3
    assert context.source_evidence.construction.limited
    assert not any(m.constructorBindingIds for m in capture.modules)


def test_forged_constructor_references_and_using_a_call_binding_are_rejected(tmp_path):
    model, context, _ = construct(tmp_path)
    capture = run(model, context)
    payload = capture.model_dump(mode='json')
    for bad in [['unknown'], capture.invocations[-1].sourceBindingIds,
                payload['modules'][0]['constructorBindingIds'] * 2]:
        payload['modules'][0]['constructorBindingIds'] = bad
        with pytest.raises(ValidationError):
            CaptureSnapshot.model_validate(payload)


@pytest.mark.parametrize('expression', ['space.Block()', '(Block if True else None)()'])
def test_descriptor_and_expression_calls_keep_location_without_claiming_direct_class(tmp_path, expression):
    source = SOURCE.replace('class Model', '''hits = []
class Namespace:
    @property
    def Block(self):
        hits.append("called")
        return Block
space = Namespace()
class Model''').replace('self.a = Block()', 'self.a = ' + expression)
    model, context, namespace = construct(tmp_path, source)
    capture = run(model, context)
    first = next(m for m in capture.modules if 'a' in m.paths)
    assert first.constructorBindingIds and not first.constructorClassVerified
    assert namespace['hits'] == (['called'] if expression == 'space.Block()' else [])


def test_dynamic_module_attribute_is_not_evaluated_again_by_class_observation(tmp_path):
    source = SOURCE.replace('class Model', '''import types
hits = []
space = types.ModuleType("layers")
def lookup(name):
    hits.append(name)
    return Block
space.__getattr__ = lookup
class Model''').replace('self.a = Block()', 'self.a = space.Block()')
    model, context, namespace = construct(tmp_path, source)
    assert namespace['hits'] == ['Block']
    assert not next(m for m in run(model, context).modules if 'a' in m.paths).constructorClassVerified
    assert namespace['hits'] == ['Block']


def test_class_proof_never_triggers_namespace_key_equality_callbacks(tmp_path):
    source = SOURCE.replace('class Model', '''import types
hits = []
class Collision:
    def __hash__(self): return hash("Block")
    def __eq__(self, other):
        hits.append("comparison")
        return False
space = types.ModuleType("layers")
space.__dict__[Collision()] = None
space.Block = Block
hits.clear()
class Model''').replace('self.a = Block()', 'self.a = space.Block()')
    namespace, context = prepared(tmp_path, source)
    namespace['create']()
    baseline = list(namespace['hits'])  # Dict probing can compare more than once.
    assert baseline
    namespace['hits'].clear()
    with context.source_evidence.construction.observe():
        model = namespace['create']()
    assert namespace['hits'] == baseline
    assert next(m for m in run(model, context).modules if 'a' in m.paths).constructorClassVerified
    assert namespace['hits'] == baseline


def test_class_change_after_initialization_invalidates_direct_class_proof(tmp_path):
    model, context, namespace = construct(tmp_path)
    assert context.source_evidence.construction.class_verified(model.a)
    class Other(namespace['Block']):
        pass
    model.a.__class__ = Other
    assert context.source_evidence.construction.binding(model.a) is not None
    assert not context.source_evidence.construction.class_verified(model.a)


def test_large_module_namespace_bounds_class_proof_without_stopping_execution(tmp_path):
    source = SOURCE.replace('class Model', '''import types
space = types.ModuleType("layers")
for i in range(8200):
    space.__dict__[str(i)] = None
space.Block = Block
class Model''').replace('self.a = Block()', 'self.a = space.Block()')
    model, context, _ = construct(tmp_path, source)
    capture = run(model, context)
    first = next(m for m in capture.modules if 'a' in m.paths)
    assert first.constructorBindingIds and not first.constructorClassVerified
    assert capture.status == 'complete'


def test_exhausted_local_lookup_cannot_fall_back_to_a_matching_global_class():
    from types import SimpleNamespace
    from services.semantic.construction import _is_direct_class
    local = {str(i): None for i in range(8200)}
    local['Ctor'] = lambda: None
    frame = SimpleNamespace(f_locals=local, f_globals={'Ctor': torch.nn.Linear},
                            f_code=SimpleNamespace(co_flags=0))
    assert not _is_direct_class(('Ctor',), frame, torch.nn.Linear)


def test_direct_class_proof_does_not_retain_a_local_class_or_its_instance(tmp_path):
    source = '''import torch
def create():
    class Local(torch.nn.Module):
        def forward(self, h): return h
    return Local()
'''
    model, context, _ = construct(tmp_path, source)
    assert context.source_evidence.construction.class_verified(model)
    reference, class_reference = weakref.ref(model), weakref.ref(type(model))
    del model
    gc.collect()
    assert reference() is None and class_reference() is None


def test_class_proof_is_optional_for_old_captures_and_requires_one_constructor_site(tmp_path):
    model, context, _ = construct(tmp_path)
    payload = run(model, context).model_dump(mode='json')
    for module in payload['modules']:
        module.pop('constructorClassVerified')
        module.pop('initializerArguments')
    assert not any(m.constructorClassVerified for m in CaptureSnapshot.model_validate(payload).modules)
    payload['modules'][0].update(constructorClassVerified=True, constructorBindingIds=[])
    with pytest.raises(ValidationError, match='one exact construction site'):
        CaptureSnapshot.model_validate(payload)
    capture = run(model, context).model_dump(mode='json')
    binding = next(b for b in capture['sourceBindings'] if b['id'] == capture['modules'][0]['constructorBindingIds'][0])
    binding['quality'] = 'ambiguous'
    with pytest.raises(ValidationError, match='exact construction binding'):
        CaptureSnapshot.model_validate(capture)

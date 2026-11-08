"""Palette deletion must accept the host's exact constructor anchor safely."""
import ast

import pytest

from services.ide import projection
from tests.source_review.test_code_apply import _apply, _apply_body, client

SOURCE = '''# tensorviz: input=2,4
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 5)
    def forward(self, x):
        # Keep the producer and return semantics.
        return self.fc(x)  # preserve return comment
'''


def inserted():
    result = _apply(SOURCE, [dict(op='add_layer', op_id='add', node_id='new_layer',
        layer_op='Dropout', params={'p': 0.5}),
        dict(op='connect', op_id='place', source_node_id='fc', target_node_id='new_layer')])
    assert result.refused == []
    assert 'return self.new_layer(self.fc(x))' in result.source
    return result.source


def body(source):
    request = _apply_body(source, [dict(op='remove_layer', op_id='delete', node_id='new_layer')])
    assignment = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == 'new_layer' for t in n.targets))
    call = assignment.value
    request['spanmap'] = {'new_layer': dict(file=request['file'], start=[call.lineno, call.col_offset],
        end=[call.end_lineno, call.end_col_offset], kind='call')}
    return request


def test_actual_palette_add_then_constructor_anchor_delete_returns_original_source(client):
    source = inserted()
    response = client.post('/api/ide/apply', json=body(source))
    assert response.status_code == 200, response.text
    assert response.json()['refused'] == []
    # Replay returned patches, as the native writer does; no file write by API.
    from tests.source_review.test_code_apply import _replay
    assert _replay(source, response.json()['edits']) == SOURCE


@pytest.mark.parametrize('change', [
    lambda s: s.replace('self.new_layer(self.fc(x))', 'self.new_layer(self.fc(x), side_effect())'),
    lambda s: s.replace('self.new_layer(self.fc(x))', 'self.new_layer(input=self.fc(x))'),
    lambda s: s.replace('self.new_layer(self.fc(x))', 'self.new_layer(*self.fc(x))'),
    lambda s: s.replace('self.new_layer(self.fc(x))', 'self.new_layer(self.fc(x), flag=side_effect())'),
    lambda s: s.replace('return self.new_layer', 'alias = self.new_layer\n        return self.new_layer'),
    lambda s: s.replace('return self.new_layer(self.fc(x))', 'return self.new_layer(self.new_layer(self.fc(x)))'),
    lambda s: s.replace('nn.Dropout(p=0.5)', 'unknown(p=0.5)'),
    lambda s: s.replace('p=0.5', 'p=side_effect()'),
    lambda s: s.replace('self.new_layer = nn.Dropout(p=0.5)', 'self.new_layer = nn.Dropout(p=0.5); side_effect()'),
    lambda s: s.replace('self.new_layer = nn.Dropout(p=0.5)', 'self.new_layer = nn.Dropout(p=0.5)  # keep constructor comment'),
    lambda s: s.replace('self.new_layer = nn.Dropout(p=0.5)', 'self.new_layer = nn.Dropout(p=0.5)\n        self.new_layer = nn.Dropout(p=0.2)'),
    lambda s: s.replace('from torch import nn', 'from torch import nn\nnn = custom'),
    lambda s: s.replace('def __init__(self):', 'def __init__(self, nn=None):'),
    lambda s: s.replace('from torch import nn', 'from torch import nn\nimport custom as nn'),
    lambda s: s.replace('p=0.5', '**options'),
    lambda s: s.replace('nn.Dropout(p=0.5)', 'nn.Dropout(  # keep this constructor comment\n            p=0.5)'),
])
def test_constructor_anchor_delete_refuses_ambiguous_or_effectful_source(client, change):
    source = change(inserted())
    response = client.post('/api/ide/apply', json=body(source))
    assert response.status_code == 200, response.text
    assert response.json()['refused']
    assert response.json()['edits'] == []


def test_constructor_anchor_delete_still_checks_source_currency(client):
    request = body(inserted())
    request['source_digest'] = projection.digest_of(SOURCE)
    assert client.post('/api/ide/apply', json=request).status_code == 409


def test_real_capture_palette_roundtrip_uses_the_host_constructor_capability(client, tmp_path):
    from tests.source_review.test_semantic_candidate import prepare
    from services.semantic.source_edits import CapturedEditResolver
    from tests.source_review.test_code_apply import _replay

    source = inserted().replace('class Net(', 'class Model(')
    _, capture = prepare(tmp_path / 'added', source)
    module = next(m for m in capture.modules if m.paths == ['new_layer'])
    invocation = next(i for i in capture.invocations if i.moduleId == module.id)
    assert invocation.status == 'completed'
    assert not [op for op in capture.operations if op.invocationId == invocation.id], 'eval Dropout remains an unchanged call'
    cap = CapturedEditResolver(capture, {'model.py': source}).constructor_literal_capabilities(invocation.id)[0]
    anchor = cap.expression.anchors[0]
    request = dict(source=source, cls='Model', source_digest=projection.digest_of(source),
        ops=[dict(op='remove_layer', op_id='delete', node_id='new_layer')])
    request['file'] = anchor.file
    request['spanmap'] = {'new_layer': dict(file=anchor.file,
        start=[anchor.startLine, anchor.startColumn], end=[anchor.endLine, anchor.endColumn])}
    response = client.post('/api/ide/apply', json=request)
    assert response.status_code == 200 and response.json()['refused'] == [], response.text
    after = _replay(source, response.json()['edits'])
    assert after == SOURCE.replace('class Net(', 'class Model(')
    _, final = prepare(tmp_path / 'deleted', after)
    def output_shape(record):
        return next(v.shape for v in record.values if v.id == record.output.id)
    assert capture.status == final.status == 'complete'
    assert output_shape(capture) == output_shape(final) == [2, 5]
    assert all('new_layer' not in m.paths for m in final.modules)


def test_identity_forward_statement_comments_are_preserved(client):
    source = inserted().replace('return self.new_layer(self.fc(x))  # preserve return comment',
        'x = self.fc(x)\n        x = self.new_layer(x)  # keep this comment\n        return x')
    response = client.post('/api/ide/apply', json=body(source))
    assert response.status_code == 200 and response.json()['refused'] == [], response.text
    from tests.source_review.test_code_apply import _replay
    after = _replay(source, response.json()['edits'])
    assert 'x = x  # keep this comment' in after
    assert 'self.new_layer' not in after


@pytest.mark.parametrize('anchor', ['operand', 'wrong_end', 'other_class'])
def test_constructor_anchor_is_exact_and_belongs_to_the_selected_model(client, anchor):
    source = inserted()
    if anchor == 'other_class':
        source += '\nclass Other(nn.Module):\n    def __init__(self):\n        self.new_layer = nn.Dropout(p=0.5)\n'
    request = body(source)
    if anchor == 'operand':
        call = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == 'fc')
        request['spanmap']['new_layer'].update(start=[call.lineno, call.col_offset], end=[call.end_lineno, call.end_col_offset])
    elif anchor == 'wrong_end':
        request['spanmap']['new_layer']['end'][1] -= 1
    else:
        assignment = ast.parse(source).body[-1].body[0].body[0]
        call = assignment.value
        request['spanmap']['new_layer'].update(start=[call.lineno, call.col_offset], end=[call.end_lineno, call.end_col_offset])
    response = client.post('/api/ide/apply', json=request)
    assert response.status_code == 200 and response.json()['refused'], response.text
    assert response.json()['edits'] == []


@pytest.mark.parametrize('case', ['wildcard', 'constructor_internal_comment'])
def test_review_import_and_constructor_comment_refusals_leave_source_unchanged(case):
    from services.ide.code_apply import apply_operations
    source = '''from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.new_layer = nn.Dropout(p=0.5)
    def forward(self, x):
        return self.new_layer(x)
'''
    if case == 'wildcard':
        source = source.replace('from torch import nn', 'from torch import nn\nfrom custom import *')
    else:
        source = source.replace('nn.Dropout(p=0.5)', 'nn.Dropout(\n            # preserve rationale\n            p=0.5\n        )')
    call = next(n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign))
    span = dict(file='model.py', start=[call.lineno, call.col_offset], end=[call.end_lineno, call.end_col_offset])
    result = apply_operations(source, {'new_layer': span}, [dict(op='remove_layer', node_id='new_layer')],
        cls='Net', file='model.py')
    assert result.refused
    assert result.edits == []
    assert result.source == source

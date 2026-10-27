"""Helper edits must appear at their real source and dataflow positions."""
import hashlib
import pytest

from services.ide.multifile import MultifileProjectionRequest
from services.ide.workspace_projection import project_workspace


SOURCE = '''from torch import nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)
    def prepare(self, x, masks=None):
        x = self.patch(x)
        # first activation
        x = x + 1
        # second activation
        return x
    def features(self, x, masks=None):
        x = self.prepare(x, masks=masks)
        return {"tokens": self.norm(x), "raw": x}
    def forward(self, *args, is_training=False, **kwargs):
        ret = self.features(*args, **kwargs)
        return self.head(ret["tokens"])
'''
AFTER = SOURCE.replace('# first activation', 'x = nn.ReLU()(x)').replace(
    '# second activation', 'x = nn.ReLU()(x)')


def project(source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    return project_workspace(MultifileProjectionRequest.model_validate({
        'protocol_version': 1, 'root': {'file': 'model.py', 'cls': 'Net', 'source_digest': digest},
        'sources': [{'file': 'model.py', 'source': source, 'digest': digest}], 'review_only': True,
    }))


def test_nested_helper_inline_activations_have_exact_spans_and_connections():
    result = project(AFTER)
    semantic = result['document']['semantic']
    relus = [node for node in semantic['nodes'] if node['op'] == 'ReLU']
    assert len(relus) == 2
    relus.sort(key=lambda node: result['spanmap'][node['node_id']]['start'])
    a, b = [node['node_id'] for node in relus]
    add = next(node['node_id'] for node in semantic['nodes'] if node['op'] == 'torch.add')
    edges = {(edge['source_node_id'], edge['target_node_id']) for edge in semantic['edges']}
    assert {('input', 'patch'), ('patch', a), (a, add), (add, b), (b, 'norm'),
            ('norm', 'head'), ('head', 'output')} <= edges
    assert (b, 'head') not in edges, 'dictionary key selection must not merge raw tokens into head'
    for node in relus:
        span = result['spanmap'][node['node_id']]
        assert AFTER.splitlines()[span['start'][0] - 1].strip() == 'x = nn.ReLU()(x)'
        assert span['kind'] == 'call'
    assert not any(node['op'] == 'ReLU' for node in project(SOURCE)['document']['semantic']['nodes'])


def test_helpers_bind_keywords_defaults_and_keep_repeated_calls_separate():
    source = '''from torch import nn
class Net(nn.Module):
    def helper(self, x, *, slope=0.2):
        return nn.LeakyReLU(slope)(x)
    def forward(self, x):
        left = self.helper(x)
        right = self.helper(left, slope=-0.1)
        return right
'''
    result = project(source)
    nodes = [n for n in result['document']['semantic']['nodes'] if n['op'] == 'LeakyReLU']
    assert len(nodes) == 2
    assert {n['params']['negative_slope'] for n in nodes} == {0.2, -0.1}
    assert len({str(n['params']['_review_scope']) for n in nodes}) == 2


def test_recursive_and_decorated_helpers_stay_visible_and_bounded():
    for helper in ['    def helper(self, x):\n        return self.helper(x)\n',
                   '    @unknown\n    def helper(self, x):\n        return nn.ReLU()(x)\n']:
        source = 'from torch import nn\nclass Net(nn.Module):\n' + helper + (
            '    def forward(self, x):\n        return self.helper(x)\n')
        nodes = project(source)['document']['semantic']['nodes']
        assert any(n['op'] == 'Python' for n in nodes)
        assert len(nodes) < 8


def test_inline_non_torch_constructor_is_not_mislabeled_as_torch():
    source = 'from torch import nn\nfrom foreign import ReLU\nclass Net(nn.Module):\n    def forward(self,x):\n        return ReLU()(x)\n'
    assert not any(n['op'] == 'ReLU' for n in project(source)['document']['semantic']['nodes'])


@pytest.mark.parametrize('call', ['self.helper(x)', 'self.helper(value=x)', 'self.helper(*(x,))', 'self.helper(**{"value": x})'])
def test_literal_argument_unpacking_and_return_containers_keep_producers(call):
    source = '''from torch import nn
class Net(nn.Module):
    def helper(self, value):
        first = nn.ReLU()(value)
        second = nn.Sigmoid()(first)
        return first, second
    def forward(self, x):
        early, late = CALL
        return early
'''.replace('CALL', call)
    semantic = project(source)['document']['semantic']
    relu = next(n['node_id'] for n in semantic['nodes'] if n['op'] == 'ReLU')
    sigmoid = next(n['node_id'] for n in semantic['nodes'] if n['op'] == 'Sigmoid')
    edges = {(e['source_node_id'], e['target_node_id']) for e in semantic['edges']}
    assert (relu, 'output') in edges and (sigmoid, 'output') not in edges


def test_refused_binding_does_not_duplicate_argument_operations():
    source = '''from torch import nn
class Net(nn.Module):
    def helper(self, x):
        return x
    def forward(self, x):
        return self.helper(nn.ReLU()(x), unexpected=True)
'''
    nodes = project(source)['document']['semantic']['nodes']
    assert sum(n['op'] == 'ReLU' for n in nodes) == 1
    assert sum(n['op'] == 'Python' for n in nodes) == 1


def test_unreachable_helper_statements_are_not_inserted_into_the_graph():
    source = '''from torch import nn
class Net(nn.Module):
    def helper(self, x):
        return nn.ReLU()(x)
        x = nn.Sigmoid()(x)
    def forward(self, x):
        return self.helper(x)
'''
    nodes = project(source)['document']['semantic']['nodes']
    assert any(n['op'] == 'ReLU' for n in nodes)
    assert not any(n['op'] == 'Sigmoid' for n in nodes)


def test_a_recognized_torch_container_subclass_has_a_reviewable_source_graph():
    source = 'from torch import nn\nclass Net(nn.ModuleList):\n    def forward(self,x):\n        for block in self:\n            x = block(x)\n        return x\n'
    assert project(source)['document']['semantic']['nodes']


def test_dynamic_dictionary_keys_do_not_duplicate_earlier_value_operations():
    source = '''from torch import nn
class Net(nn.Module):
    def forward(self, x, key):
        return {"first": nn.ReLU()(x), key: nn.Sigmoid()(x)}
'''
    nodes = project(source)['document']['semantic']['nodes']
    assert sum(n['op'] == 'ReLU' for n in nodes) == 1
    assert sum(n['op'] == 'Sigmoid' for n in nodes) == 1


def test_unordered_set_arguments_do_not_prove_helper_parameter_bindings():
    source = '''from torch import nn
class Net(nn.Module):
    def helper(self, x, y):
        return nn.ReLU()(x)
    def forward(self, x, y):
        return self.helper(*{x, y})
'''
    nodes = project(source)['document']['semantic']['nodes']
    assert not any(n['op'] == 'ReLU' for n in nodes)
    assert any(n['op'] == 'Python' for n in nodes)

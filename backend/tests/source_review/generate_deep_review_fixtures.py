"""Regenerate real dynamic-child and literal-container renderer review evidence.

Run from backend: python -m tests.source_review.generate_deep_review_fixtures
"""
import json
import tempfile
from pathlib import Path
from tests.source_review.test_deep_review_declarations import SOURCE, LITERAL_SOURCE, project, document, setup


def host(result):
    return dict(graphId=result['graph_id'], revision=result['revision'], cls=result['cls'], file='model.py',
        sourceDigest=result['source_digest'], traced=False, greyed=False, viewRestored=False,
        document=result['document'], spanmap=result['spanmap'], dependencies=result['dependencies'], diagnostics=result['diagnostics'])


def main():
    output = Path(__file__).resolve().parents[3] / 'canvas/src/test/fixtures/canvas/deepDeclarationReview.json'
    fixtures = {}
    expanded = LITERAL_SOURCE.replace('self.blocks = nn.ModuleList([Block() for _ in range(2)])', 'self.first = Block()\n        self.last = Block()').replace('for block in self.blocks:\n            x = block(x)\n        return x', 'return self.last(self.first(x))')
    direct = SOURCE.split('class Model')[0].replace('class Block', 'class Model')
    with tempfile.TemporaryDirectory() as directory:
        for name, source, candidate in [
            ('expanded', expanded, expanded.replace('nn.ReLU(), nn.ReLU()', 'nn.GELU(), nn.ReLU()')),
            ('equivalent', direct, direct.replace('nn.Linear(4, 4, bias=False)', 'nn.Linear(in_features=4, out_features=4, bias=False)')),
            ('dynamic', SOURCE, SOURCE.replace('bias=False', 'bias=True')),
            ('static', LITERAL_SOURCE.replace('nn.ModuleList([Block() for _ in range(2)])', 'nn.Sequential(Block(), Block())'),
             LITERAL_SOURCE.replace('nn.ModuleList([Block() for _ in range(2)])', 'nn.Sequential(Block(), Block())').replace('nn.ReLU(), nn.ReLU()', 'nn.GELU(), nn.ReLU()')),
            ('literal', LITERAL_SOURCE, LITERAL_SOURCE.replace('nn.ReLU(), nn.ReLU()', 'nn.GELU(), nn.ReLU()')),
        ]:
            before, after = host(project(source)), host(project(candidate))
            capture, _ = setup(Path(directory).resolve(), source)
            captured = {**before, 'traced': True, 'document': document(capture, source).model_dump(mode='json', by_alias=True)}
            fixtures[name] = dict(before=before, after=after, capture=captured)
            if name == 'literal':
                fixtures[name]['inserted'] = host(project(source.replace('nn.ReLU(), nn.ReLU()', 'nn.GELU(), nn.ReLU(), nn.ReLU()')))
                fixtures[name]['reordered'] = host(project(source.replace('nn.Linear(4, 4), nn.ReLU()', 'nn.ReLU(), nn.Linear(4, 4)')))
    output.write_text(json.dumps(fixtures, indent=2) + '\n')


if __name__ == '__main__':
    main()

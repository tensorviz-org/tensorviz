"""Regenerate renderer fixtures from real captures and production preflight.

Run from backend: python -m tests.source_review.generate_nested_assignment_fixtures
Capture ids are intentionally regenerated; review the fixture diff with its tests.
"""
import json
import tempfile
from pathlib import Path

from pytest import MonkeyPatch

from tests.graph.test_ide_multifile_projection import client
from tests.source_review.test_nested_source_assignments import (
    NESTED_SOURCE, host_capture_projections, multifile_capture_projections, nested_capture_projections,
)


def host_payload(result):
    return dict(graphId=result['graph_id'], revision=result['revision'], cls=result['cls'], file='model.py',
                sourceDigest=result['source_digest'], traced=result['traced'], greyed=False, viewRestored=False,
                document=result['document'], spanmap=result['spanmap'], dependencies=result['dependencies'],
                diagnostics=result['diagnostics'])


def main():
    output = Path(__file__).resolve().parents[3] / 'canvas/src/test/fixtures/canvas'
    with tempfile.TemporaryDirectory() as temporary, MonkeyPatch.context() as patch:
        root = Path(temporary).resolve()
        api = client.__wrapped__(root)
        results = nested_capture_projections(api)
        proposed = NESTED_SOURCE.replace('self.blocks = nn.Sequential(Block())',
            'self.blocks = nn.Sequential(Block())\n        self.blocks[0].relu = nn.GELU()')
        results['validated'] = nested_capture_projections(api, proposed)['capture']
        (output / 'nestedAssignmentCapture.json').write_text(json.dumps(
            {name: host_payload(result) for name, result in results.items()}, indent=2) + '\n')
        containers = {}
        for kind, constructor, call, target in [
            ('ModuleList', 'nn.ModuleList([Block()])', 'self.blocks[0](x)', 'self.blocks[0].relu'),
            ('ModuleDict', 'nn.ModuleDict({"head": Block()})', 'self.blocks["head"](x)', 'self.blocks["head"].relu'),
        ]:
            result = nested_capture_projections(api, NESTED_SOURCE.replace('nn.Sequential(Block())', constructor)
                                                .replace('self.blocks(x)', call), target)
            containers[kind] = {side: result[side]['document'] for side in ('capture', 'after')}
        for kind in ('Multifile', 'PartialMultifile'):
            directory = root / kind
            directory.mkdir()
            capture, after, *_ = multifile_capture_projections(api, patch, directory if kind == 'PartialMultifile' else None)
            containers[kind] = dict(capture=capture['document'], after=after['document'])
        directory = root / 'host'
        directory.mkdir()
        capture, after = host_capture_projections(api, directory)
        containers['HostPartial'] = dict(capture=capture['document'], after=after['document'])
        (output / 'nestedContainerCaptures.json').write_text(json.dumps(containers, indent=2) + '\n')


if __name__ == '__main__':
    main()

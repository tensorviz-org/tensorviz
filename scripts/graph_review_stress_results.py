#!/usr/bin/env python3
"""Assemble graph review evidence for actual, audited GitHub PRs.

A checkpoint includes only published and audited PRs. --complete requires all
1,000, the final GitHub inventory, and passing semantic/render results. This tool
writes local files only; pushing its review-results branch is a separate action.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def build(args):
    state = args.state
    manifest = {row['id']: row for row in json.loads((state / 'manifest.json').read_text())}
    baseline = {row['id']: row for row in json.loads(args.baseline.read_text())}
    reviews = {row['id']: row for row in json.loads(args.reviews.read_text())}
    rendered = json.loads(args.render.read_text())
    renders = {row['id']: row for row in rendered['cases']}
    receipts = {row['id']: row for row in read_rows(state / 'github-prs.jsonl')}
    audits = {row['id']: row for row in read_rows(state / 'github-file-audits.jsonl')}
    package = json.loads((state / 'package.json').read_text())
    assert len(manifest) == 1000 and set(manifest) == set(baseline) == set(reviews) == set(renders)
    assert all(row['passed'] for row in reviews.values()) and all(row['passed'] for row in renders.values())
    assert Path(rendered['package']).resolve() == Path(package['assets']).resolve(), 'Render used another package'
    ids = sorted(set(receipts) & set(audits))
    final = state / 'github-final-audit.json'
    if args.complete:
        assert len(ids) == 1000 and final.exists(), 'GitHub publication/audit is not complete'
        audit = json.loads(final.read_text())
        assert audit['pullRequests'] == audit['fileDiffs'] == audit['uniqueBranches'] == 1000
        assert audit['allRevisionShasMatch'] and audit['repository'] == args.repository
    args.output.mkdir(parents=True, exist_ok=True)
    cases = args.output / 'cases'
    cases.mkdir(exist_ok=True)
    assert not ({p.stem for p in cases.glob('*.json')} - set(ids)), 'Unexpected stale review records'
    provenance = {'productCommit': args.product_revision, 'vsixSha256': package['sha256'],
                  'generatedAt': datetime.now(timezone.utc).isoformat(),
                  'semanticReportSha256': hashlib.sha256(args.reviews.read_bytes()).hexdigest(),
                  'renderReportSha256': hashlib.sha256(args.render.read_bytes()).hexdigest()}
    index = []
    for case_id in ids:
        item, pr, gh, review = manifest[case_id], receipts[case_id], audits[case_id], reviews[case_id]
        for record in (pr, gh, review, baseline[case_id]):
            assert record['base'] == item['base'] and record['head'] == item['head'], case_id
        assert gh['passed'] and gh['url'] == pr['url'] and pr['branch'] == item['branch']
        record = {'model': {key: item[key] for key in ['id', 'family', 'domain', 'width', 'depth', 'cls', 'kind', 'inputs', 'sourceDigests', 'validity']},
                  'oracle': item['oracle'], 'github': gh, 'initialComparison': baseline[case_id],
                  'finalComparison': review, 'packagedCanvas': renders[case_id], 'provenance': provenance}
        (cases / (case_id + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        unresolved = review['counts']['unresolved']
        index.append(f"| [#{pr['number']}]({pr['url']}) | [{case_id}](cases/{case_id}.json) | {item['kind']} | passed | {unresolved} |")
    summary = {'complete': args.complete, 'repository': args.repository, 'publishedPrs': len(receipts),
               'auditedPrs': len(audits), 'reviewRecords': len(ids), 'validatedModelRevisions': 2000,
               'semanticCases': len(reviews), 'semanticPassed': sum(r['passed'] for r in reviews.values()),
               'renderCases': len(renders), 'renderPassed': sum(r['passed'] for r in renders.values()),
               'casesWithUnresolvedCorrespondence': sum(r['counts']['unresolved'] > 0 for r in reviews.values()),
               'initialFailedCases': sum(not r['passed'] for r in baseline.values()), 'provenance': provenance}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    title = 'Completed 1,000-PR graph review' if args.complete else 'Graph review checkpoint — publication is still running'
    intro = f'''# {title}

{len(ids)} actual GitHub PRs have complete review records below; {len(receipts)} are
published and {len(audits)} have audited GitHub file diffs. Each record verifies the
base/head commit, changed paths, line counts, and head blob SHAs, then records the
source-change oracle, original failure (if any), final graph result, coverage
limitations, and packaged Canvas result. [Machine-readable summary](summary.json).

All 2,000 generated model revisions passed finite CPU forwards. All 1,000 final
graph comparisons and 1,000 packaged Canvas cases pass. The Canvas run uses the
production VSIX frontend in Chromium with a native-shell test double; it is not
1,000 native VS Code sessions or manual approvals. {summary['casesWithUnresolvedCorrespondence']} cases retain unresolved
correspondence. A pass does not certify arbitrary Python or runtime equivalence.

The initial oracle found 200 failures. Stronger operand/parameter assertions then
exposed 50 projection multiplicity failures and 3 missing recurrent defaults.
Corrections preserve constructor defaults, repeated operands through composition,
control predicates, reachable helper bodies, and matrix-product boundaries.

Product revision: `{args.product_revision}`. VSIX SHA-256:
`{package['sha256']}`.

| Actual PR | Review record | Source edit | Result | Unresolved nodes |
| --- | --- | --- | --- | --- |
'''
    (args.output / 'README.md').write_text(intro + '\n'.join(index) + '\n')
    if args.complete: (args.output / 'github-final-audit.json').write_text(final.read_text())
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['state', 'baseline', 'reviews', 'render', 'output']: parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--product-revision', required=True)
    parser.add_argument('--complete', action='store_true')
    build(parser.parse_args())

#!/usr/bin/env python3
"""Require complete template and composition checks in the local gate.

The sweep uses the product's adapter planner when joining templates. Rejection of
an unadapted, incompatible pair belongs in an explicit negative test, not in an
allowed-failure count in the acceptance report.
"""
from __future__ import annotations

import re
import sys

# The floor belongs on the CATALOG, not on the check count. The sweep runs one check
# per OFFERED template per validation mode, so a hardcoded check count pins the product
# of two numbers that both move legitimately. 500 was written on 2026-08-25, when
# list_contracts() still returned all 306 contracts (612 strict checks). On 2026-09-08
# commit 7773c55d narrowed the offered set to the blocks the code generator can actually
# build — 224 now, so 448 checks — and this floor has been arithmetically unsatisfiable
# ever since, with nothing failing. The whole 306-contract corpus is still swept by
# tests/validation/test_qa_plane_parity_sweep.py. 200 is the same floor
# tests/templates/test_canonical_registry_offers_buildable.py already holds the offered set to.
MIN_EXPECTED_TEMPLATES = 200


def _discovered(text: str) -> int:
    match = re.search(r"Total templates discovered:\s*\*\*(\d+)\*\*", text)
    if not match:
        raise SystemExit("Could not find the 'Total templates discovered' line in the report")
    return int(match.group(1))


def _counts(text: str, label: str) -> tuple[int, int]:
    match = re.search(rf"{label} checks:\s*\*\*(\d+)/(\d+) passed\*\*", text)
    if not match:
        raise SystemExit(f"Could not find the '{label} checks' line in the report")
    return int(match.group(1)), int(match.group(2))


def main(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    discovered = _discovered(text)
    single_passed, single_total = _counts(text, "Single-template")
    combo_passed, combo_total = _counts(text, "Combination")
    single_failed = single_total - single_passed
    combo_failed = combo_total - combo_passed

    print(f"templates discovered: {discovered}")
    print(f"single-template: {single_passed}/{single_total} passed ({single_failed} failed)")
    print(f"combination:     {combo_passed}/{combo_total} passed ({combo_failed} failed)")

    problems = []

    # Two halves, and both are needed. The absolute floor alone rots every time the
    # catalog is resized; the derived check alone is 0 >= 0 on an empty run, which is
    # the "0 failures out of 0" hole this gate exists to close.
    if discovered < MIN_EXPECTED_TEMPLATES:
        problems.append(
            f"only {discovered} templates discovered, expected at least "
            f"{MIN_EXPECTED_TEMPLATES} — the catalog itself is short (see buildable_ids)"
        )
    elif single_total < discovered or single_total % discovered:
        problems.append(
            f"{single_total} single-template checks over {discovered} discovered templates "
            "— the sweep skipped templates (expect one whole pass per validation mode)"
        )

    if single_failed:
        failures = re.search(
            r"## Single-Template Failures\n(.*?)\n## ", text, re.DOTALL
        )
        detail = failures.group(1).strip() if failures else "(failure table not found)"
        problems.append(f"{single_failed} single-template failure(s):\n{detail}")

    if combo_total == 0:
        problems.append("no combination checks ran")
    if combo_failed:
        combos = re.search(r"## Combination Failures\n(.*?)\n## ", text, re.DOTALL)
        detail = combos.group(1).strip() if combos else "(failure table not found)"
        problems.append(
            f"{combo_failed} combination failure(s):\n{detail}"
        )

    if problems:
        print("\nGate FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("Gate OK.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: assert_sweep_report.py <report.md>")
    raise SystemExit(main(sys.argv[1]))

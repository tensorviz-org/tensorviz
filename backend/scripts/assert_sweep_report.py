#!/usr/bin/env python3
"""Assert on a template-validation sweep report.

The gate used to `grep -q 'Single-template checks: **612/612 passed**'`. That
pins a *denominator*: adding or removing one template broke the gate even though
nothing regressed, and the obvious fix — editing the magic number — silently
turns the check into a no-op, because "N/N passed" is true for any N including
zero. It also never looked at the "Combination checks" line, which has read
0/2 for as long as the report has existed.

So this asserts the two things that actually matter:

  * zero single-template failures, whatever the total is; and
  * the total is not suspiciously small, so an empty or truncated sweep cannot
    pass as "0 failures out of 0".

Combination failures are reported loudly and compared against a documented
baseline. They are aspp_block -> autoregressive_decoder_block in both modes, and
they are NOT a Conv2d head fed a non-4D tensor as this docstring used to say: the
gate runs with --disable-adapters, which chains two independently-seeded blocks
with no shape reconciliation, so ASPP's 15-channel output meets a head declared
at in_channels=3. Validation is right to reject that graph, and the product's own
chaining path (plan_adapter_transition) bridges it — --enable-adapters reports
2/2. Failing the gate on them
would leave main red on a harness-only mode; ignoring them silently is how the
old grep let 0/2 sit unnoticed. Printing them and holding the line at the known
count does neither.
"""
from __future__ import annotations

import re
import sys

# aspp_block -> autoregressive_decoder_block, in both standard and strict mode.
# Reaching 0 means the gate's
# combination pass chains templates the way the product does (adapters on) —
# a reviewer's call about what the gate measures, not a number to edit; raising
# it needs a register entry and a reviewer too.
KNOWN_COMBINATION_FAILURES = 2

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

    if combo_failed > KNOWN_COMBINATION_FAILURES:
        combos = re.search(r"## Combination Failures\n(.*?)\n## ", text, re.DOTALL)
        detail = combos.group(1).strip() if combos else "(failure table not found)"
        problems.append(
            f"{combo_failed} combination failures, above the known {KNOWN_COMBINATION_FAILURES}:\n{detail}"
        )
    elif combo_failed:
        print(
            f"NOTE: {combo_failed} known combination failure(s), at the documented "
            f"baseline of {KNOWN_COMBINATION_FAILURES}."
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

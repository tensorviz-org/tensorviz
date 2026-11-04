# Compact model context sync — 2026-11-04

The user requested these changes in `tensorviz-org/tensorviz`, default `main`,
with commit identity `tensorviz <tensorviz.app@gmail.com>`. Destination base:
`252d6d806e457cd47bfeff529bbd476b9c446e12`. Source: extension PR #196, commits
`b5cacbcb2`, `a72df8aad` and `15aa33ac0` (merged as `84b49c243`).

## Import map and boundaries

Import only shared model context, its MCP registration/freshness check, Canvas
attachment/budgets and native/ACP guidance, corresponding tests, model-workflow
skill and installed/native acceptance drivers. The QA driver maps `pytorch-viz`
to `canvas`; the CPU integration test maps `mainbackend` to `backend`.
Adapt `McpGraphService` to its existing telemetry-free constructor and keep the
destination’s native instructions and browser options. Update the consolidated
MCP, Canvas and plugin guides rather than importing upstream feature-history docs.

Dependency pins, package identities/versions, release scripts and CI stay local.
No analytics, latency profiling, training-step capture, hosted website code,
benchmark runner or earlier performance claims are imported. This is a source
push, with `[skip ci]`; no workflow or destination release is requested.

## Verification

Passed shared-host build and all 1,145 graph-host tests, extension compile and
1,970 unit tests, 3,335 Canvas tests, 72 plugin unit tests, protocol/semantic/
validation checks, 18 boundary/tooling tests and documentation links.

Real CPU capture plus MCP passed both distinct and tied-weight cases: unique
root parameter counts 18 and 12, each called layer 6, input [2,3], output [2,2],
and no calls invented for the unused registration. Claude Code 2.1.308 passed
installation, 28-tool discovery, actual capture, a compact 15-parameter Linear
packet with [2,4] input and [2,3] output, and uninstall in a disposable profile.

The first extension run caught a missed ACP guidance hook during adaptation;
adding the shared guidance to the destination's existing instructions fixed it,
and the complete unit suite then passed. The first live checks used Torch 2.7.1,
which lacks `torch._dynamo.package` required by the destination's existing worker.
Those failed runs are not acceptance. Both live checks passed with the compatible
Torch 2.15.0 environment; no backend import or test expectation was changed.

No backend Python source or Canvas source changed. Full backend pytest, the wider
extension integration suite and a newly installed destination VSIX were not run.
The source extension's published VSIX acceptance is recorded separately in its
feature record; it does not certify a new artifact from this repository.

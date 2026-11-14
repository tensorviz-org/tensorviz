# Service setup follow-up at d1c463c

This is a bounded source check of `d1c463c37863d89c2ccc8e37ec75858de47d194a` on macOS 26.4 (25E246), arm64. It is separate from the published plugin and VSIX acceptance results, and from the earlier source GUI checks at `eae4dd5d012c090d31ca1149891287ea01febc6c`. No desktop UI or production release gate was tested in this follow-up.

Between those source commits, [PR #74](https://github.com/tensorviz-org/tensorviz/pull/74), merged as `f046c32d1188d84564576773d9164af64af0c9f1`, is the only runtime change. PRs #72 and #75 change documentation. The pinned Python runtime manifest and service/capture requirement files are unchanged across that range; this change selects the existing service manifest in more provisioning paths.

Fresh service provisioning and additive repair now select `requirements-standalone.txt`. The provisioning stamp hashes that manifest alone. The first-install disclosure says model frameworks are not installed, and the packaging script requires the service manifest. Capture still uses `requirements-capture.txt`, which declares Torch, and its own interpreter/probe. The execution paths support the intended separation. Historical comments in `requirements-cpu.txt` and one runtime stamp comment remain stale.

| Local check | Observed result |
| --- | --- |
| Independent root/extension npm installs and TypeScript compilation | Passed; Node 26.5.0, npm 11.17.0. Pending package lifecycle script approvals were left unchanged. |
| Extension runtime, lifecycle and preflight suites | 58 passed. Fixture-based provisioning, consent, cancellation, additive repair and preservation checks; no real desktop approval. |
| Backend service provisioning and real static service | 5 passed in Python 3.12.13 arm64 without Torch. Real loopback service health, source discovery, projection and setup exercised with missing/broken framework imports. |
| Capture separation suite | 5 passed in Python 3.12.13 arm64 with Torch 2.14.1. Includes a real synthetic factory-model capture and checks prohibiting tool-server imports. |
| Broader environment/installer suite, initial attempt | 109 passed, 2 failed. The original failed run is retained. See the explanations below. |
| Internal capture dependency probes | Missing Torch returned structured `MISSING_DEPENDENCIES`; healthy Torch returned version 2.14.1. Both had empty stderr. These are internal probes, not end-to-end MCP error UI acceptance. |

The broader suite contains one real capture-worker test, so running all of it with the no-Torch interpreter was a harness mistake. That specific test passed when rerun with the capture interpreter. The other failure is a checkout-path-sensitive assertion: it expects exactly 64 traceback frames, while the implementation also limits traceback text to 12,000 characters. This deep checkout reaches the text limit after 63 frame markers. The same test fails at its original path with the capture interpreter and passes from a SHA-verified, identical test-file copy at a short temporary path. No repository tests or product code were changed to make it pass. There is no claim that the original broader suite passed.

`source-test-receipts.json` records the original commands/results. `failure-followups.json` records those targeted reruns and the exact test-file hash. The reusable `check_current_source.py` now selects the capture interpreter for the mixed environment suite; its explicit `--reproduce-original-no-torch-suite` option retains the original harness selection. `check_failure_followups.py` reproduces the path comparison without editing the repository.

No provider turns, auth copying, changes to normal client profiles, permission grants, workflows, merges or releases occurred. All installer fixtures use task-owned temporary environments and local wheels with package-index access disabled. Existing test interpreters were used without package changes. Raw logs and temporary profiles remain private; the receipts contain versions, counts, source hashes and raw-log hashes only. This follow-up did not download a fresh pinned CPython or perform a full real managed-service installation; actual VSIX packaging and graphical acceptance are reported separately.

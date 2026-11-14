# macOS backend and model-capture acceptance

Source tested: `eae4dd5d012c090d31ca1149891287ea01febc6c` (main snapshot after PRs #69–#71). This is source acceptance for the 0.1.45 development work, not acceptance of a published 0.1.45 package.

Host: macOS 26.4, arm64. Both test interpreters were independently executed and reported **arm64, Python 3.12.13**. The no-Torch virtualenv contains the standalone-service requirements and pytest, with no importable `torch`. The capture virtualenv contains **PyTorch 2.14.1**, **transformers 5.23.0**, and **diffusers 0.43.0**. Exact package inventories are in the JSON results and lockfiles. Existing user environments were preserved; none were removed.

## Observed results

- **Passed:** 7 standalone/extension backend tests. Real loopback servers answered static health, discovery, projection, parsing, templates, structural validation and code-generation routes without Torch. Missing and simulated broken-native Torch paths produced explicit `pytorch_unavailable` responses. Static routes did not execute the synthetic model.
- **Passed:** 54 capture, saved-factory, input-proposal, stale-source, intentional model-error, structural-diff and Torch-cache tests.
- **Passed:** 110 environment-diagnostic tests. One capture-only test was initially assigned to the no-Torch environment and failed because it requires Torch. Rerunning that test in the capture environment passed. The initial command and failure remain recorded as a harness setup error, not a product bug. The runner now assigns that test to the correct environment.
- **Passed:** 4 tiny random transformer-family capture tests. BERT produced 108 graph nodes, DINOv2 98, and ViT 99 with complete capture. T5 produced 339 nodes with **partial** capture because its default `EncoderDecoderCache` is opaque; the test explicitly verifies that limitation.
- **Passed:** 25 conditional-UNet/input tests; one full-size default case was deselected. The two saved-profile tiny UNet captures produced 885 and 912 graph nodes and output `[1,4,16,16]`, with complete capture.
- **Passed with a recorded capture limitation:** the separate tiny GPT-2 script ran its forward pass and produced 86 graph nodes, 74 edges, and `last_hidden_state` shape `[3,5,16]`. Capture is **partial** because the unchanged default `DynamicCache` is opaque.
- The capture probe in the absent-Torch environment returned `{ok:false, code:MISSING_DEPENDENCIES, message:"Missing capture library: torch."}`. The healthy environment returned `{ok:true, torch:"2.14.1", cuda:null}`. The helper's process exit is zero in both cases; the `ok` field distinguishes them.

The suites produced 201 passing test outcomes plus the GPT-2 script outcome. No new source defect requiring a patch was reproduced in this scope. The existing no-Torch extension fix from [PR #69](https://github.com/tensorviz-org/tensorviz/pull/69) was exercised at the pinned main SHA.

## Scope limits

These are backend/service and worker-level tests, not desktop UI acceptance or signed-in client testing. Models use randomly initialized, deliberately small configurations with synthetic inputs, no downloaded weights and no personal/private data. They do not complete issue #179's full manual model matrix: full pretrained DINOv2, Stable Diffusion pipelines, F5-AEC, Nemotron, and Karpathy repositories were not tested here. The T5/GPT-2 opaque-cache limitations remain visible. MCP/ACP graph diff and manual GUI edits are outside this backend evidence.

## Reproduction

Use a separate checkout of the exact SHA and new test environments. Install only into these disposable environments. For native Apple silicon, choose an arm64 Python 3.12 explicitly and verify `platform.machine()`; an existing `python3.12` may instead be x86_64 under Rosetta.

```sh
uv venv --python /path/to/arm64-python3.12 no-torch-env
uv venv --python /path/to/arm64-python3.12 capture-env
uv pip install --python no-torch-env/bin/python -r requirements-no-torch.lock.txt
uv pip install --python capture-env/bin/python -r requirements-capture.lock.txt
python3 run_backend_checks.py --source /path/to/pinned-checkout --output results --no-torch-python no-torch-env/bin/python --capture-python capture-env/bin/python
python3 run_library_checks.py --source /path/to/pinned-checkout --output results --python capture-env/bin/python
capture-env/bin/python capture_gpt2.py --source /path/to/pinned-checkout --output results
```

The two test runners set disposable HOME/cache/temp directories and pass a small explicit environment, without GitHub/provider credentials or SSH-agent variables. Library captures set offline flags and use no model weights. Runtime subprocesses bind only loopback. These are isolation conventions, not an OS security sandbox; run only reviewed source. The scripts install nothing and publish nothing. Raw logs stay in `results/private`; reviewed summaries are separate.

The rerun command layout separates the one capture-dependent environment test; the recorded first run used a single broader suite, as shown in `backend-results.json`. No product source files were modified.

## Published 0.1.44 VSIX baseline

The actual backend files from the installed 0.1.44 VSIX were also launched with the arm64 Python 3.12.13 no-Torch environment. **Failed:** the extension's `server_v2:app` exited before its HTTP health endpoint became available. The supplemental `standalone_server.py` source entry bundled inside that VSIX also exited before reporting an address. Both fail through `core/op_registry.py:_build_torch_nn` importing `torch.nn` during module initialization (`ModuleNotFoundError: No module named 'torch'`). Static-route requests are **blocked by startup failure**, not passed. This is an exposed-service startup check, separate from an internal `probe-capture` command. It is also separate from the native plugin package's compiled runtime.

The pinned main source passes the equivalent no-Torch service tests and contains the already-merged fix in PR #69; no duplicate patch was created. `installed-vsix-service-results.json` includes hashes of the actual tested entry files. `probe_installed_backend.py` reproduces the service check with a supplied installed-backend directory and no-Torch interpreter. It uses a fresh per-launch token, the normal extension IDE mode, private HOME/state paths, and loopback-only sockets; no system security settings change.

## Pure broken-Torch fixture

A separate arm64 Python 3.12.13 environment was prepared with all capture dependencies and synthetic `torch==2.14.1` distribution metadata. Its `torch/__init__.py` deliberately raises an `OSError`; it is **not real PyTorch** and no healthy environment was changed. This isolates an import failure from missing Pydantic/NumPy dependencies.

Both the installed 0.1.44 helper and pinned main source reported `missing: []`, `incompatible: []`, and Torch metadata `2.14.1` during `capture_environment.py inspect`. Their subsequent `probe-capture` returned `ok:false`, `CAPTURE_RUNTIME_UNSUPPORTED`, and “The capture worker could not load (OSError). Choose a compatible environment.” This is distinct from the absent-Torch `MISSING_DEPENDENCIES` response. The high-level helper names the exception type, but does not expose the synthetic native-error text. This probe is not a GUI acceptance claim.

Reproduction uses a **new** virtualenv:

```sh
uv venv --python /path/to/arm64-python3.12 broken-torch-env
uv pip install --python broken-torch-env/bin/python -r requirements-no-torch.lock.txt
python3 create_broken_torch_fixture.py --env broken-torch-env
broken-torch-env/bin/python -I -B /path/to/backend/capture_environment.py inspect
broken-torch-env/bin/python -I -B /path/to/backend/capture_environment.py probe-capture
```

The fixture creator refuses an interpreter outside the requested virtualenv and refuses to overwrite an installed Torch package. `pure-broken-torch-preflight.json` records both compared helpers.

## Environment-row status bug fixed in draft PR

Actual source-built extension testing showed a broken interpreter’s row claiming **Ready for this model** immediately beside the correctly reported capture-worker error. The panel used installed-package metadata without considering the known worker failure. [Draft PR #76](https://github.com/tensorviz-org/tensorviz/pull/76), commit `b2cb6827ebb90c261d030ab64f3fbd98113edf3e`, now marks that saved interpreter **Capture check failed** with error styling. Other interpreters retain their own status, and a fresh host verdict can restore normal readiness. No protocol or execution behavior changes.

The new regression failed before the fix; all 54 environment-panel tests passed afterward. `git diff --check` passed. An extra `tsc --noEmit` attempt was blocked by TypeScript 4.9.5 rejecting newer syntax in installed tldraw/Zod declarations. The patched VSIX was not rebuilt/reinstalled. No workflows, merges, or releases ran.

The fix is based on `f046c32d1188d84564576773d9164af64af0c9f1`, which adds #72 glibc docs and #74 service-runtime provisioning changes after the original `eae4dd5` test snapshot. The panel itself was byte-identical at both commits. Exact commands and outcomes are in `environment-status-fix.json`.

## Actual published extension 0.1.45 backend follow-up

The SHA-verified **published** `tensorviz-vscode-0.1.45.vsix` (5,641,948 bytes,
SHA-256 `edb17c4d891f9a6ec531c86d8f0771778bedc6f50cfe7ecf19db5789aace1990`)
was tested separately after publication. Its release source is
`eae4dd5d012c090d31ca1149891287ea01febc6c`. This run loads backend code only
from that archive; the pinned source checkout supplies the HTTP acceptance test
harness, not application modules.

All eight bounded checks passed in native arm64 Python 3.12.13:

- Both actual bundled service entry points started and served static projection,
  discovery and setup in a genuinely no-Torch environment. Import blockers
  additionally exercised missing and simulated broken framework imports.
- The extension service served parsing, templates, structural validation and
  code generation; eight Torch-required requests returned explicit HTTP 503
  `pytorch_unavailable` diagnostics with the expected missing/broken reason.
- Capture-helper probes returned structured `MISSING_DEPENDENCIES` for no Torch
  and `CAPTURE_RUNTIME_UNSUPPORTED` for the separate synthetic broken-import
  fixture. Healthy Torch 2.14.1 passed the probe.
- The packaged worker captured a real synthetic residual MLP: 211 parameters,
  input `[2,4]`, output `[2,3]`, complete semantic capture, 27 nodes and 20 edges.
  Graph inspection was programmatic; no graphical acceptance is implied.

This confirms the previously merged no-Torch service fix in the newly published
VSIX. It does not erase the retained 0.1.44 startup failure or establish
extension activation, managed runtime installation, plugin capture, or UI behavior.
All test service children exited; raw logs and synthetic files remain private.
Exact environment versions, entry-point hashes and per-check outcomes are in
[published-0.1.45-backend-results.json](published-0.1.45-backend-results.json).

Reproduction uses existing isolated interpreters and a checkout pinned to the
recorded source SHA. Substitute only local paths; the command does not install
packages, alter client settings or publish anything:

```sh
<NO_TORCH_PYTHON> run_published_vsix_backend.py \
  --vsix <RELEASES>/tensorviz-vscode-0.1.45.vsix \
  --sha256 edb17c4d891f9a6ec531c86d8f0771778bedc6f50cfe7ecf19db5789aace1990 \
  --version 0.1.45 \
  --release-source-sha eae4dd5d012c090d31ca1149891287ea01febc6c \
  --test-source <SOURCE_EAE4DD5> \
  --no-torch-python <NO_TORCH_PYTHON> \
  --broken-torch-python <SYNTHETIC_BROKEN_TORCH_PYTHON> \
  --capture-python <HEALTHY_TORCH_PYTHON> \
  --output <PRIVATE_OUTPUT>
```

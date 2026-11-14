# Documentation and release provenance evidence

> Historical inspection before plugin 0.1.45 was published at 16:32:18 UTC. See the report README and publication-0.1.45 evidence for the later state. The glibc correction remains valid; the newly published target list needs a broader documentation update. Current status: PR72 was externally merged at16:46:09Z; deployment is unverified. Both extension and plugin0.1.45 were later published; see release-authoritative-recheck.json.

Inspected on 2026-11-14. The initial inspection made read-only HTTPS/GitHub API
requests and created these local evidence files. A separately authorized docs
correction then installed docs dependencies in an isolated worktree, ran local
documentation checks, and opened the draft PR recorded below. No application
tests, settings or credential changes, desktop session use, graph publication,
workflow dispatch, deployment, merge, or release was performed by this subtask.

## Results

- The bounded live URL check returned **HTTP 200 for 56 of 56 endpoints/assets**.
  It found no missing IDs for the linked anchors it checked.
- Three documentation graph illustrations (`explore-graph.png`, `follow-code.png`,
  `review-changes.png`) and both theme logos were reachable. This proves asset
  delivery, not visual quality or a successful user journey.
- Open VSX reported extension **0.1.44**. GitHub extension and plugin 0.1.44
  metadata both identify source **`f285a383f628af23ac4992970eeab3b006b1f0d2`**.
  Plugin export/tag commit is **`6fff2d501cbe037e2d90b252ee484a446c37bd61`**.
- Source head **`eae4dd5d012c090d31ca1149891287ea01febc6c`** and its live source-build
  documentation identify development version **0.1.45**. This is distinct from
  the published 0.1.44 artifacts. No 0.1.45 release or tag was returned by these
  lookups at the recorded time.
- This subtask fetched the small plugin release manifest and macOS runtime
  metadata. It did **not** download/install or independently hash the runtime
  archive or VSIX; metadata agreement is not installed-artifact acceptance.

## Confirmed documentation drift

1. **The main installation guides retain the previous Linux glibc floor.**
   The [live installation guide](https://docs.tensorviz.dev/install.html) and
   [agent guide](https://docs.tensorviz.dev/agent-plugin.html) say `glibc >=2.35`.
   The published [0.1.44 plugin catalog](https://github.com/tensorviz-org/tensorviz-plugins/tree/v0.1.44)
   and its `SUPPORT.md` say `glibc >=2.31`, consistent with the
   [0.1.44 extension release note](https://github.com/tensorviz-org/tensorviz/releases/tag/v0.1.44).
   In the inspected source revision, the stale current-install copy is at:
   `README.md:12`, `docs/.vitepress/theme/InstallOptions.vue:69`,
   `docs/agent-plugin.md:13`, `docs/environment-recovery.md:84`,
   `docs/install.md:19`, `docs/quickstart.md:5`, and
   `docs/reference/agent-plugin.md:82`. These should be aligned to the verified
   published floor. Preserve the historical statement about 0.1.43 in
   `docs/linux-ssh-runtime.md:59` and the Linux ARM64 source target's 2.35 floor.
   No Linux runtime execution was performed in this Mac documentation subtask.
2. **The homepage omits the published Linux plugin platform.** The
   [homepage](https://tensorviz.dev/) plugin installation note still describes
   an Apple Silicon macOS preview only. The published 0.1.44 catalog also serves
   Linux x64. Update website copy against published platform metadata; do not
   announce source-only 0.1.45 platforms as released.

At inspection, `tensorviz` had no open PRs. `tensorviz-plugins` had draft
[PR #3](https://github.com/tensorviz-org/tensorviz-plugins/pull/3), which documents
historical cold-start recovery/cache isolation and is not a duplicate of the
glibc-floor correction. `tensorViz-website` had
[PR #15](https://github.com/tensorviz-org/tensorViz-website/pull/15), concerning a
Model Explorer addition rather than plugin installation copy.

## Source and release distinctions

- [PR #69](https://github.com/tensorviz-org/tensorviz/pull/69), merged as
  `44fe97077e389517cd15000cf8dedad57ce15b71`, changes extension-service startup
  without PyTorch. Its merged source behavior must not be assumed present in
  published 0.1.44 (`f285a383...`).
- [PR #70](https://github.com/tensorviz-org/tensorviz/pull/70), merged as
  `4b575e7ff2955f8d04f70af0a05d90e17631c406`, prepares 0.1.45 distribution and
  six-platform/migration gates. Recorded PR test claims were read, not rerun.
- [PR #71](https://github.com/tensorviz-org/tensorviz/pull/71), merged as
  `eae4dd5d012c090d31ca1149891287ea01febc6c`, fixes the release TLS fixture and
  retains startup diagnostics. Its description explicitly separates a published
  0.1.44 diagnostic from the required 0.1.45 production gates.
- The extension's `latest` tag remains an immutable download anchor at an older
  source commit. Its release description points at 0.1.44 and explicitly says to
  use the numbered release for source. Do not treat that tag's commit as the
  source of the current attached VSIX.

## Reproduction and limits

Run from this evidence directory with a Python whose existing TLS trust verifies
the public sites:

```sh
python3 check_urls.py --output live-url-check.json
```

The successful run used CPython 3.14.6 / OpenSSL 3.6.3. A separate available
CPython 3.14.6 / OpenSSL 3.5.7 runtime failed certificate verification before HTTP
on all 14 seeds; see `live-url-check-tls-blocked.json`. Certificate validation
remained enabled in both runs. No trust store, proxy, or network setting was
changed. Those TLS failures are an execution-environment limitation, not proof
that the sites are broken.

The script uses unauthenticated GET requests, strips redirect query strings, and
stores URL/status/hash/selected public metadata only. It does not save cookies,
headers, raw HTML, local paths, raw exception messages, or private session data.
The scope covers seed installation/docs pages, their directly linked docs pages,
and images in the seed pages. It does not crawl all pages or execute JavaScript,
copy buttons, install commands, sign-in, graph upload, or client UI flows.

`checklist-coverage.json` maps all 13 items in
[release checklist #179](https://github.com/tensorviz-org/tensorViz-extension/issues/179)
to this subtask's actual evidence and remaining checks. No runtime/GUI/model item
is marked passed, and results from other agents are not assumed.

Evidence files: `live-url-check.json`, `live-url-check-tls-blocked.json`,
`release-provenance.json`, `source-doc-platform-lines.json`, and
`checklist-coverage.json`. `release-status-recheck.json` records the final release
API recheck; use its timestamp for the publication-status observation.

## Authorized draft correction

[Draft PR #72](https://github.com/tensorviz-org/tensorviz/pull/72) corrects the
seven current-install glibc-floor references, with no other product changes.
Branch: `codex/fix-published-plugin-glibc-docs`.
Commit: `3c95274e16114c1289b25c2958a238085037b91f`, based on the inspected
`eae4dd5d012c090d31ca1149891287ea01febc6c` source revision.
The commit includes `[skip ci]` and uses the repository's tensorviz Git identity.

Validation passed on macOS arm64 with Node 26.5.0, npm 11.17.0 and Python 3.14.6:

- `python3 tooling/docs/check_links.py`: all links resolve across 83 files.
- `DOCS_BASE=/tensorviz/ npm run build --prefix docs`: passed.
- `npm run build --prefix docs`: passed.
- Each build verified 56 pages, 3,617 links/anchors, 55 source-edit links and
  56 Markdown downloads.
- `git diff --check`: passed; worktree clean after committing.

Node 22 was not available locally; the exact documented Node 22 build was not
run. Npm reported pending install-script approval for esbuild and fsevents; both
documentation builds succeeded without changing the script-approval policy.
VitePress reported existing direct-eval and chunk-size warnings. No product
runtime or Linux installation test was run for this text-only correction.
The PR was draft and its check list was empty at initial inspection. A later API check confirmed external merge at16:46:09Z (commit c0d54ea043337bc04ebdcc6c0b4f56a3d54ded59); deployment was not verified. The separate website compatibility-copy finding remains unfixed.

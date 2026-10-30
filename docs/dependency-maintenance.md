# Dependency maintenance

Routine Dependabot version PRs are paused with `open-pull-requests-limit: 0` in
each configured package location. Maintainers can review compatible updates as
one batch, validate the combined tree locally, and squash it into one commit.
Security alerts and security-update PRs are separate repository settings; this
configuration does not enable them. Check those settings before public launch.

## October 2026 consolidation

The batch incorporates these existing PRs:

- Requests 2.34.2 ([#7](https://github.com/tensorviz-org/tensorviz/pull/7)),
  MarkupSafe 3.0.4 ([#10](https://github.com/tensorviz-org/tensorviz/pull/10)), and
  python-dotenv 1.2.4 ([#13](https://github.com/tensorviz-org/tensorviz/pull/13)):
  maintained backend fixes, including dotenv parsing of empty values, BOMs, and
  Windows backslashes. The project and standalone minimums include the dotenv fix.
- Mocha 12.0.3 in both hosts
  ([#12](https://github.com/tensorviz-org/tensorviz/pull/12),
  [#22](https://github.com/tensorviz-org/tensorviz/pull/22)) and
  openapi-typescript 7.13.0
  ([#14](https://github.com/tensorviz-org/tensorviz/pull/14)):
  remove vulnerable development dependency chains reported by npm audit.
  Both OpenAPI generator installations stay aligned. The generator now prints
  the returned TypeScript AST and preserves Python's required/optional fields
  with `defaultNonNullable: false`. Generated type files are regenerated; the
  wire schemas and runtime validation policy are unchanged.
- VSCE 4.0.0 ([#15](https://github.com/tensorviz-org/tensorviz/pull/15)) and
  the native plugin packager 6.24.0
  ([#20](https://github.com/tensorviz-org/tensorviz/pull/20)):
  refresh packaging dependencies while retaining the documented Node 22 runtime.
- MCP extensions 0.2.1
  ([#25](https://github.com/tensorviz-org/tensorviz/pull/25)):
  current plugin extension SDK fixes. The separate MCP Apps protocol remains on
  its existing version in both Canvas and plugin.
- YAML 2.9.1 ([#27](https://github.com/tensorviz-org/tensorviz/pull/27)):
  fixes recursive merge-alias handling. Its lockfile retains the existing Canvas
  overrides and clean-install dependency tree.

The following upgrades are deferred rather than combined into the launch tree:

- CI action major versions (#4–#6): the existing CI pins remain usable.
- A platform-specific torchvision `+cpu` pin (#8): macOS wheels have no `+cpu`
  suffix; the current CPU-index setup already selects the appropriate wheel.
- Raising the ONNX minimum (#9): fresh installs already resolve the current
  version; no feature here requires dropping the older supported minimum.
- Node 26 type packages (#11, #19, #24) and newer VS Code declarations (#16):
  declaration versions must follow supported hosts, not the newest package.
- Testing Library major migrations (#17, #23), MCP Apps 2 (#18), TypeScript 7
  (#21), tldraw 5.5 (#26), and Playwright 1.64 (#28): these require separate
  compatibility work or provide no required launch fix. In particular, the
  renderer and its assets must move together, and TypeScript 7 breaks the
  existing compiler-API boundary checks.

The consolidation does not add model profiling, GPU optimization, or new product
features. Remaining legacy development dependency advisories are not a claim
that this entire tree has a clean security audit.

The cache changelog example uses inline code so its `<time>` placeholder does
not become an unterminated HTML tag in the newly merged documentation site.

## Validation

Use the checks in [local setup](local-setup.md#checks), including generated
contract checks, both host suites, Canvas tests/build, backend tests, and local
package verification. Run `npm ci` against every changed lockfile; do not accept
an incremental install as proof that a clean checkout can install.

## Canvas agent transport dependencies

The source sync adds the ACP SDK to the editor host and AI SDK transport packages to Canvas. Destination security overrides and existing release identities remain. AI SDK upstream notice metadata is retained under `plugin/licenses/` for archive notice generation; notices are still assembled from the actual bundle. The local VSIX is a development artifact and does not establish renderer distribution permission.

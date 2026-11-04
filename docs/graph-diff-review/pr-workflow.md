# Direct PR and code-diff review

The entry points are **Graph diff** on an open native Python diff and **Open PR**
in Canvas or the Command Palette. Review discovers
models from the compared snapshots without requiring a live model or running code.

`POST /api/ide/review/index` indexes supplied, digest-checked Python sources. It
reuses the multi-file resolver's module inheritance rules, returns declarations
and transitive local file dependencies, and retains source spans. Syntax errors,
unknown model inheritance, dynamic imports, and truncated inventories stay visible.
Unique identical declaration bodies with identical import context provide rename
candidates; topology and declaration order do not prove identity. File dependency
discovery is conservative: selecting an affected model can legitimately yield no
structural changes.

The route reads source text only and uses the projection contract's path, count,
per-file and total-byte limits. Empty snapshots are valid for model additions and
deletions. `backend/tests/source_review/test_graph_review_index.py` verifies these boundaries.

## Source loading

`githubReview.ts` accepts GitHub.com PR URLs, reads the PR and its merge base, and
pins the base and head source trees, including fork heads. Each downloaded blob
must match its Git object digest before entering the repository-scoped private
extension cache. Existing VS Code GitHub authentication or GitHub CLI credentials
are reused; otherwise VS Code requests sign-in. Credentials never enter source
URLs, logs, or cached files. The cache stores source text, not executable checkouts.

`editorGraphReview.ts` captures both documents in the selected native diff before
Canvas receives focus. Built-in Git HEAD/index/working comparisons and GitHub PR
diffs retain the displayed revisions. Unsaved Python buffers are captured exactly;
staged content is read by immutable blob ID. A mismatch between provider metadata
and displayed text refuses the review. Unknown providers can compare their two
documents with explicit incomplete-dependency coverage. No branch, index, or
working file is changed.

`reviewSnapshots.ts` discovers affected models across both revisions, including
unchanged model files that import a changed helper. Project boundaries come from
each source tree. Source budgets and unavailable files produce partial coverage;
an absent model is not called added/deleted when the other inventory is incomplete.
Tests cover pinned fork/merge-base loading, source corruption, path and size limits,
cache reuse, dirty/index isolation, active-diff identity, and project boundaries.

Models under nested project folders retain their declarations and captured context
when review runs from the Git checkout root. Directory aliases use the same file
identity for unsaved buffers, navigation and stale-review checks. Conflicting text
in two open aliases must be resolved before comparison; source-file symlinks remain
excluded. See [project boundaries](../graph-opening.md#models-inside-a-larger-workspace).

## Reviewer workflow

- In VS Code's Python code diff, use **Graph diff** in the editor toolbar. The
  displayed base and head are captured before Canvas takes focus.
- A connected agent can call `open_graph_diff` for Git HEAD versus current files,
  including unsaved Python buffers. It opens this same review and returns the
  selected model, changed files and source coverage notes without waiting for
  Canvas rendering confirmation. Optional `file` and `cls` select the
  model; no capture or execution run ids are needed. See [MCP tools](../mcp-tools.md#show-local-changes-in-the-graph).
- From empty Canvas, use **Open PR**, or run **tensorViz: Open PR**. Paste a
  GitHub.com PR link directly into the picker, or choose one of the repository's
  100 most recently updated open PRs. Pasting remains available while the list
  loads or if listing fails. Closed/merged PR links work too.
- The first affected model opens automatically. **Affected model** switches
  among the discovered models, including models affected by shared local helpers.
  Selecting a model opens its class in the native code diff. Within this explicit
  review, moving the cursor into another discovered class follows that model;
  clicking graph nodes reveals the corresponding code revision and line.
- Before / Diff / After retain the existing canvas layout. Whole-model additions
  and deletions are explicit. Unique declaration/import fingerprints can establish
  a rename; ambiguous matches remain unresolved. A failed projection keeps the
  selector usable and exposes an incomplete comparison with the failure detail.
- A PR stays pinned. An update check runs on open and once per minute; **PR updated**
  offers explicit refresh. Working snapshot edits similarly require refresh.
  Refresh retains the selected model when its identity still exists.

The older **Review model changes** command remains available for advanced local
revision selection. Direct entry does not require Open as Graph, constructor
inputs, runtime execution, a checkout, or an external browser. Closing review
returns to the retained live model or empty Canvas.

## Limits

GitHub.com is supported; enterprise hosts are not yet wired. Discovery is AST-only
and uses each project's existing 256-file, 1-MiB-per-file and 8-MiB-total limits.
Working snapshots additionally cap the repository at 8,192 files / 64 MB. Coverage
notes remain visible for limits, syntax errors, unresolved model inheritance and
imports. Unknown editor providers supply only their displayed documents and are
marked partial. No claim is made about behavioral equivalence, arbitrary Python
edits, library internals, or runtime shapes. Remote SSH uses the workspace extension
host.

Current Git added/deleted-file diffs preserve the surrounding repository when
VS Code uses an empty-tree URI for the absent file. A deleted clean editor buffer
does not resurrect source; unsaved edits remain part of the working snapshot.
Empty-tree comparisons against unrelated historical/custom revisions are refused
with an explicit message because the sentinel alone does not identify a complete
repository revision. A staged first-ever commit with no HEAD still requires the
PR route after publication; it is not treated as an invented Git base.

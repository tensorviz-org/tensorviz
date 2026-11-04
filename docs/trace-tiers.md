# Trace tiers

`tensorviz.traceTier` controls graph captures without an explicit saved execution
setup. The resource setting accepts `real`, `fake`, and `auto`; `real` is the
default. Saved execution profiles and exact candidate checks use real tensors.

- `real` constructs and runs the model with real tensors, including branches that
  depend on tensor values.
- `fake` constructs the model, inputs, and forward under one `FakeTensorMode`.
  Parameters and intermediate tensors carry metadata without allocating their
  data. Module imports run outside that construction mode, so module-level
  constants retain their ordinary behavior.
- `auto` starts each model with real tensors. If that capture reports at least
  150 million parameters, later captures of that model use fake tensors. Smaller
  models continue using real tensors. The policy remembers models by workspace,
  file, and class, retains at most 256 entries, and resets when the window reloads.

Canonical fake captures retain graph, shape, dtype, storage, and source evidence;
values are unavailable. A recognized fake-tensor limitation, such as a value read,
dynamic output shape, unsupported operator, or lazy parameter, triggers one real
retry using the same prepared capture. Other failures are reported directly.
After such a fallback, `auto` remembers that the model needs real tensors.
Cancellation or a superseding request prevents a stale fallback from launching.

Explicit saved execution and candidate requests continue to require the real tier.
The execution cache never serves or stores a fake-tier run.

## Repeated blocks

`TENSORVIZ_BLOCK_TEMPLATES=1` enables within-capture templates for eligible repeated
blocks in canonical fake captures. The first matching block runs live; subsequent
matching blocks can reuse its recorded structure with fresh identities and fake
outputs. This is off by default and does not affect real captures or legacy fake
traces. `TENSORVIZ_BLOCK_TEMPLATES=shadow` compares the stamped structure against a
live recording. The variable is read when the worker imports the trace module.

Blocks with unsupported state changes, random operations, value reads, hooks,
exceptions, or incomplete recorder evidence remain ineligible. Limit and identity
checks guard reuse. Optional `block_templates` diagnostics appear only when this
mode is active; fake-tensor limitation diagnostics appear only for the relevant
failure. Ordinary real results omit both fields.

Code focus shows its Parameter source selector only when a parameter is selected.
Its scope and opacity controls remain available for other selections.

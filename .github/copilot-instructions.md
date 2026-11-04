# tensorViz

Read [AGENTS.md](../AGENTS.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).

`extension/` owns the editor, source review, and the local service. `canvas/` is
the graph UI. `backend/` parses, projects, and captures. Shared protocol lives
in `packages/contracts`. Do not add profiling, product analytics, or secrets.

The editor buffer is authoritative, including unsaved changes. Static source
inspection does not import the model. Browser edits are proposals: the extension
checks source freshness, opens native diffs, and applies accepted changes through
one undoable editor edit. Keep launch credentials and provider keys in the host.

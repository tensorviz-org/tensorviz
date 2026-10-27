# Shared contracts

Canonical TypeScript wire contracts and capture readers live in `src/`. Python
owns semantic and validation policy in `backend/services/*/contracts.py`;
the generated TypeScript and runtime schema must not be edited by hand.

Run `npm ci` at the repository root, then `npm run build:shared`. Node consumers
use `@tensorviz/contracts/<module>`. Release bundlers select the
`tensorviz-source` export condition so source inventories retain the original
tracked files. Compiled declarations and JavaScript are local build outputs.

The frontend retains checked copies for its independent compiler. Run
`npm run protocol:sync` after a wire/helper change and `npm run protocol:check`
to detect drift. Use the root `semantic:generate`, `validation:generate` and
corresponding `:check` commands for Python-owned contracts. Those checks need
the backend environment:

```sh
PYTHON=./venv/bin/python npm run semantic:check
PYTHON=./venv/bin/python npm run validation:check
```

This package cannot import application code, platform APIs or third-party
libraries. `npm run check:boundaries` enforces that rule, including type imports.
Layout of the repository is in the [README](../../README.md#layout).

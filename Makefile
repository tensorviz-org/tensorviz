# Local verification for the tensorViz extension and plugin.
# CI is .github/workflows/ci.yml. make gate is the wider local set.

export KMP_DUPLICATE_LIB_OK := TRUE

# `python` on a developer machine is usually NOT the env that has torch — the gate used
# to die at `import torch` with no hint about what to do. So the default is now the
# first interpreter that can actually import it: an active virtualenv, then the
# checkout's own ./venv, then python3, then python. check-python still turns a miss
# into one actionable line. Override on the command line:
#     make gate PYTHON=./venv/bin/python
PYTHON ?= $(shell \
  for candidate in "$$VIRTUAL_ENV/bin/python" ./venv/bin/python python3 python; do \
    if [ -n "$$candidate" ] && command -v "$$candidate" >/dev/null 2>&1 \
       && "$$candidate" -c 'import torch' >/dev/null 2>&1; then echo "$$candidate"; exit 0; fi; \
  done; echo python)

# Every backend recipe changes into backend/ first, so a RELATIVE interpreter path
# stops resolving the moment it does — and check-python runs in the repo root, so it
# reported "python OK" and the very next line died with "No such file or directory".
# `make gate PYTHON=./venv/bin/python`, the invocation docs/local-setup.md recommends,
# never worked. A path is made absolute here; a bare name is left alone, because that
# is a PATH lookup and survives the cd.
# Let Python resolve its executable; make's abspath splits paths containing spaces.
PYTHON_ABS := $(shell "$(PYTHON)" -c 'import sys; print(sys.executable)' 2>/dev/null)

# Where the gate keeps things it builds for itself. Ignored by git; delete it freely.
GATE_DIR := .gate
PROVIDER_FREE_VENV := $(GATE_DIR)/provider-free

.PHONY: gate gate-offline gate-quick hooks shared-check \
        backend-gate backend-tests backend-sweep backend-provider-free \
        frontend-gate frontend-guards frontend-tests frontend-build docs-links \
        extension-gate extension-compile extension-webview extension-unit \
        extension-interpreter extension-electron extension-package \
        plugin-gate plugin-unit plugin-capture plugin-package plugin-release \
        e2e e2e-auto e2e-smoke \
        check-python check-extension-deps

## Backend, frontend, extension and browser checks. Run this before opening a PR.
gate: shared-check docs-links backend-gate frontend-gate extension-gate plugin-gate e2e-auto

## Everything except the browser specs, which need a live stack and a Playwright browser
gate-offline: shared-check docs-links backend-gate frontend-gate extension-gate plugin-gate

## The fast legs only — what the pre-push hook runs. NOT a substitute for `make gate`:
## it leaves out the sweep, the provider-free smoke, the .vsix build and the browser
## specs, each of which has caught something the fast legs did not.
gate-quick: shared-check docs-links backend-tests frontend-tests frontend-build extension-unit plugin-unit

shared-check:
	npm run test:shared
	npm run check:boundaries
	npm run protocol:check

## Plugin checks use the same explicit Python as the backend checks.
plugin-gate: plugin-unit plugin-capture plugin-package

plugin-unit:
	cd plugin && npm test

plugin-capture: check-python
	cd plugin && TENSORVIZ_TEST_PYTHON="$(PYTHON_ABS)" npm run test:capture

plugin-package:
	cd plugin && npm run package

## Requires installed Codex and Claude CLIs (or their documented test overrides).
## Run against the extracted archive; never change the developer's live profiles.
plugin-release: plugin-package check-python
	cd plugin && TENSORVIZ_TEST_PYTHON="$(PYTHON_ABS)" npm run test:release

## Install the pre-push hook: the affected legs run before every push
hooks:
	git config core.hooksPath .githooks
	@echo "pre-push hook installed (git config core.hooksPath .githooks)."
	@echo "Skip it for one push with --no-verify; remove it with: git config --unset core.hooksPath"

## Fail with the exact invocation to use, rather than an import traceback
check-python:
	@"$(PYTHON)" -c 'import torch' >/dev/null 2>&1 || { \
	  echo ""; \
	  echo "  '$(PYTHON)' cannot import torch, so the backend gate cannot run."; \
	  echo ""; \
	  echo "  $(PYTHON) is $$(command -v "$(PYTHON)" 2>/dev/null || echo 'not on PATH')"; \
	  echo ""; \
	  echo "  Pass an interpreter that has the backend deps installed, e.g.:"; \
	  echo ""; \
	  echo "      make $(MAKECMDGOALS) PYTHON=./venv/bin/python"; \
	  echo ""; \
	  echo "  To create one, see docs/local-setup.md."; \
	  echo ""; \
	  exit 1; \
	}
	@echo "python OK: $(PYTHON) ($$("$(PYTHON)" -c 'import sys,torch; print(sys.version.split()[0], "torch", torch.__version__)'))"

## Backend: unit suite, strict sweep, and the provider-free runtime smoke
backend-gate: backend-tests backend-sweep backend-provider-free

BACKEND_SUITE ?=

## CI ran this as seven shards; that was to bound a standard runner's memory, not for
## coverage — the sharding is index-based over the same file list. One process locally
## is the same tests. If it OOMs on your machine, shard it the way CI did:
##     rg --files tests | rg '/test_.*\.py$$' | sort | awk 'NR%7==1' | xargs $(PYTHON) -m pytest -q
backend-tests: check-python
	cd backend && "$(PYTHON_ABS)" -m pytest tests/$(BACKEND_SUITE) -q

## The sweep asserts ZERO failures, not a pinned "612/612" string. Pinning the
## denominator broke the gate whenever a template was added, and editing the
## magic number silently turned the check into a no-op ("N/N passed" is true for
## any N, including zero). assert_sweep_report.py also reports the Combination
## checks line, which the old grep never looked at.
backend-sweep: check-python
	cd backend && "$(PYTHON_ABS)" scripts/run_template_validation_sweep.py \
		--disable-adapters --strict --max-combinations 1 \
		--out reports/gate-single-current.md
	cd backend && "$(PYTHON_ABS)" scripts/assert_sweep_report.py reports/gate-single-current.md

## The backend really serves HTTP and really traces a model with NO AI SDKs installed.
##
## It needs its own interpreter, and that is the whole point: the script refuses to run
## where openai / pydantic_ai / langgraph / langchain_core / opentelemetry are importable,
## and a normal development venv has all five. So the gate builds one venv from
## requirements-project.txt (the extension's own runtime list) and caches it under
## $(GATE_DIR). First run installs torch and takes a few minutes; later runs are instant.
## Delete $(PROVIDER_FREE_VENV) to rebuild it after requirements-project.txt changes.
backend-provider-free:
	@if [ ! -x "$(PROVIDER_FREE_VENV)/bin/python" ]; then \
	  echo "building the provider-free environment in $(PROVIDER_FREE_VENV) (first run only)"; \
	  "$(PYTHON_ABS)" -m venv "$(PROVIDER_FREE_VENV)" || exit 1; \
	  "$(PROVIDER_FREE_VENV)/bin/python" -m pip install --quiet --upgrade pip || exit 1; \
	  "$(PROVIDER_FREE_VENV)/bin/python" -m pip install --quiet \
	    torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu || exit 1; \
	  "$(PROVIDER_FREE_VENV)/bin/python" -m pip install --quiet \
	    -r backend/requirements-project.txt || exit 1; \
	fi
	"$(PROVIDER_FREE_VENV)/bin/python" backend/scripts/check_provider_free_runtime.py

## Frontend: the whole jest estate, then the production build as the type gate
frontend-gate: frontend-tests frontend-build

# Browser host-fixture tests load extension/out/test/bootstrap.js on clean checkouts.
frontend-tests: extension-compile
	cd canvas && npm run test:ci

## The type gate. A standalone `tsc -p tsconfig.json` cannot run here at all —
## TypeScript is pinned at 4.9.5 and tldraw 5's shipped .d.ts uses `const` type
## parameters — but CRA's checker filters node_modules out and does check every
## file under src/. So the build IS the type check; there is no shorter one.
frontend-build:
	cd canvas && CI=true npm run build

## The three fast guard suites on their own (flags, error copy, interaction audit)
frontend-guards:
	cd canvas && npm run test:ui2:guards

## Fail with the exact invocation to use, rather than a tsc module-resolution wall
check-extension-deps:
	@test -d extension/node_modules || { \
	  echo ""; \
	  echo "  extension/node_modules is missing, so the extension gate cannot run."; \
	  echo ""; \
	  echo "  Install it once:"; \
	  echo ""; \
	  echo "      cd extension && npm ci"; \
	  echo ""; \
	  echo "  Needs Node 22 (the version in .nvmrc)."; \
	  echo ""; \
	  exit 1; \
	}

## VS Code extension, including real interpreter and native-host tests.
extension-gate: extension-compile extension-webview extension-unit \
                extension-interpreter extension-electron extension-package

## Kept separate from extension-package even though vsce recompiles via vscode:prepublish:
## a bare tsc failure names the file and line, the same failure wrapped in vsce's npm output
## does not.
extension-compile: check-extension-deps
	cd extension && npm run compile

## Five of the Electron suites assert extension/media/app/index.html exists and fail
## without it, and a .vsix built without it installs fine and opens an empty panel.
extension-webview: check-extension-deps
	cd extension && npm run build:webview

## The pure seams: config env injection, interpreter precedence, port binding, checkout
## resolution. No display, no VS Code download, no interpreter.
extension-unit: check-extension-deps
	cd extension && npm run test:unit

## Real Python workers and MCP evidence checks run separately from Node-only tests.
extension-interpreter: check-extension-deps check-python
	cd extension && TENSORVIZ_TEST_PYTHON="$(PYTHON_ABS)" npm run test:integration

## The real extension host: downloads a VS Code build on first run (~300 MB, cached in
## extension/.vscode-test) and drives the backend, the worker and the browser session for
## real. Run one suite while debugging with TENSORVIZ_TEST_ONLY=<suite>.
extension-electron: check-extension-deps check-python extension-webview
	cd extension && npm run stage:backend
	cd extension && TENSORVIZ_TEST_PYTHON="$(PYTHON_ABS)" npm run test:electron

## Build the .vsix, then assert it carries the two things whose absence is invisible:
## a `main` that is not inside it installs fine and answers "command not found", and a
## missing canvas installs fine and opens an empty panel.
extension-package: check-extension-deps
	cd extension && npm run package
	@cd extension && unzip -l tensorviz-*.vsix | grep -q 'extension/out/extension.js' \
	  || { echo "the .vsix does not contain extension/out/extension.js" >&2; exit 1; }
	@cd extension && unzip -l tensorviz-*.vsix | grep -q 'extension/media/app/index.html' \
	  || { echo "the .vsix does not contain the built canvas" >&2; exit 1; }
	@echo "vsix OK: entry point and canvas both present"

## Every relative link in README.md and docs/ resolves
docs-links:
	python3 docs/check_links.py

## Browser specs against a stack you started yourself (backend :8765, frontend :3000).
## playwright.config.ts lives in canvas/, so running `npx playwright test` from the
## repo root reports "No tests found" — which in a log reads as "nothing to do" rather
## than "you are in the wrong directory". These targets cd for you, and run-e2e.mjs fails
## the run if zero tests executed.
e2e:
	cd canvas && npm run test:e2e:substantive

## The same specs, but bringing the stack up and taking it down for you — what the
## "browser specs against a live stack" job did. Use this one.
e2e-auto:
	"$(PYTHON)" --version >/dev/null 2>&1 || { echo "no interpreter for the backend" >&2; exit 1; }
	PYTHON="$(PYTHON_ABS)" scripts/e2e-stack.sh

e2e-smoke:
	cd canvas && npm run test:e2e:ui2:smoke:auto

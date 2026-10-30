# Reuse a capture

Execution caching is **on by default**. An eligible identical CPU run can reuse an earlier result in the current editor window. Python briefly checks the plan and its dependencies; the model does not run again.

## Reuse a result

1. Use a saved CPU profile in `tensorviz.toml` with a model seed and `deterministic = true`:

   ```toml
   # Inside your existing [profiles.<name>] section
   device = "cpu"
   seed = 0
   deterministic = true
   ```

   The model seed is separate from an input generator's seed.
2. Choose **Run model**, review the saved setup, then **Run**.
3. Repeat without changing the saved sources, configuration, inputs, environment, interpreter, or backend. The toolbar labels an eligible reused result.

Saving a model or tracked dependency also validates the open model by default. Set `tensorviz.traceOnSave` to `false` if you want to run manually.

## Choose a cache mode

Open Settings and search for `tensorviz.executionCache.mode`.

| Mode | What happens |
| --- | --- |
| `on` (default) | Reuses an eligible matching result after a Python safety check |
| `off` | Always executes the model |
| `shadow` | Executes every run and records whether reuse would have been possible |

The mode is read for the run's own file, so folders in a multi-root workspace can opt out separately.

## Run fresh or clear results

Open execution details from the **CPU / Reused** badge, choose **Run fresh**, then confirm **Run** to execute the model again.

Run **tensorViz: Clear Execution Cache** to discard stored results. They stay in memory for the current window and are never written to disk.

<details>
<summary>When reuse is refused and how results are checked</summary>

Agent MCP runs and reviewed-candidate runs always execute. A run that writes files, starts subprocesses, opens sockets, loads unsupported native code, or reads, lists, or checks untracked resources cannot be stored. Environment variables and tracked dependencies must still match.

Installed package files used by a stored result are fingerprinted and rechecked before serving. A waiting single-use worker also checks preloaded package files before execution. Changed packages trigger a fresh POSIX worker; other platforms refuse reuse of that stale evidence.

The window setting owns the shared store's memory limit and lifetime, with per-result and count limits. Closing the window disposes its worker and store. Reuse preserves current editor document versions and source freshness, but it is not new runtime evidence.

</details>

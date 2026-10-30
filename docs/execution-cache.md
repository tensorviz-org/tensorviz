# Execution cache

`tensorviz.executionCache.mode` defaults to `on`. Set it to `off` to always execute the model, or `shadow` to check reuse while still executing. The setting is read for the run's own file, so individual folders in a multi-root workspace can opt out. The window setting owns the shared store's memory limit and lifetime.

Reuse requires identical saved sources, configuration, inputs, seed, environment variables, interpreter and backend on CPU. Python still briefly resolves and checks the plan; the model is not run again. Agent MCP and reviewed-candidate runs always execute. The UI labels reused results, and **Run fresh** bypasses reuse.

A run that writes files, starts subprocesses, opens sockets, loads unsupported native code, or reads/lists/checks untracked resources cannot be stored. Installed package files used by a stored result are fingerprinted and rechecked before serving. A waiting single-use worker also checks preloaded package files before execution; changed packages trigger a fresh POSIX worker, while other platforms refuse reuse of that stale evidence.

Results stay in memory for the window, bounded by the configured size and per-result/count limits. **tensorViz: Clear Execution Cache** drops them. Closing the window disposes its worker and store. No cache entry is written to disk. Cache reuse preserves source freshness and current editor document versions; it is not new runtime evidence.

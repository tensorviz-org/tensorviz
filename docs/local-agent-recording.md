# Local Canvas agent recording

For an explicitly started local QA session, set `TENSORVIZ_AGENT_RECORDING_DIR` to a new absolute output directory before launching the editor. Recording is off when the variable is unset. The host writes bounded JSONL events with private file permissions, shows recording failures, and does not upload anything or change agent permissions. Recordings can contain model source and chat text; keep them outside Git.

`node tooling/ml-agent-recording.cjs` provides `start`, `stop`, `status`, and `report` commands. Run it without arguments to see the required absolute paths. Start launches the chosen editor with an isolated profile and extensions directory. Stop writes a stop marker for the extension to observe. Reports describe locally observed host events and requested model settings; they do not prove provider execution timing, model identity or task quality. Incomplete and truncated journals remain explicit.

Synthetic source/shape oracle generators live in `tooling/ml_*oracle*.py`. Run a generator with the development Python interpreter and `--help` for its output options. Checked-in fixtures record their source/capture provenance and conditional shape evidence; they are not installed-app acceptance receipts.

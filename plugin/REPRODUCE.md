# Reproduce the bounded plugin checks

Use installed Codex and Claude CLIs, already-downloaded verified catalog/runtime
artifacts, and a new evidence directory. The recorded CLI versions are in
`published-0.1.44-provenance.json`. These helpers create disposable client
profiles/state beneath the evidence directory. They do not sign in, send a
provider/model turn, or grant model-execution permission. Raw logs stay under
`private/`; review any new receipt before copying it into a report.

```sh
python3 check_cli.py --evidence /path/to/new-evidence --catalog /path/to/extracted-0.1.44-catalog --label published-0.1.44
python3 check_upgrade.py --evidence /path/to/new-upgrade-evidence --older /path/to/extracted-0.1.43-catalog --newer /path/to/extracted-0.1.44-catalog --local-only
python3 check_client_mcp.py --evidence /path/to/new-mcp-evidence --catalog /path/to/extracted-0.1.44-catalog --archive /path/to/tensorviz-runtime-0.1.44-darwin-arm64.tar.gz --label published-0.1.44
python3 check_native.py --evidence /path/to/new-native-evidence --runtime /path/to/verified-extracted-native-runtime --label published-0.1.44
```

The native directory must contain the verified `tensorviz-host` executable.
The client bridge check uses the verified archive as a warm cache. These commands
do not reproduce a cold client-handshake measurement or accepted model capture.
For a different published version, use that release's verified catalog/runtime
and its matching label; keep results separate from the 0.1.44 baseline.

For the internal dependency probe, supply the directory containing the compiled
capture helper and two existing isolated interpreters:

```sh
python3 check_capture_probe.py --evidence /path/to/new-probe-evidence --backend /path/to/compiled-backend --no-torch-python /path/to/no-torch-env/bin/python --capture-python /path/to/capture-env/bin/python --label published-0.1.44
```

This checks the helper entry point, not the user-facing permission/capture flow.
Source-development client checks use `check_client_mcp.py --source` with a
separately built source catalog and no `--archive`. Preserve its exact source
revision and archive hash, and label it separately from published/native
acceptance.

No package archives, environments, profiles or raw protocols belong on the
report branch. These process/profile conventions are not an OS sandbox; use
reviewed code and artifacts.

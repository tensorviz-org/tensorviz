#!/usr/bin/env python3
"""Reproduce the recorded published VSIX CLI lifecycle in disposable profiles.

Supply already-downloaded artifacts. SHA-256 and package identities are pinned.
Without --execute, validate artifacts and print the plan only. This script does
not download, sign in, change normal editor settings, run GUI acceptance, enable
permissions, publish, or run a GitHub workflow. It is not an OS sandbox.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import shutil
import subprocess
import tempfile
import time
import zipfile

EXTENSION = "tensorviz.tensorviz-vscode"
ARTIFACTS = {
    "0.1.43": "521d0c753e921d29ce820cd4ef3fd0fa5fcb3b58295f297826c0b0c90b3c1139",
    "0.1.44": "cb56724877d9211d4e000ade2277062b7564589a2f9808065371f4f2e4234aa8",
    "0.1.45": "edb17c4d891f9a6ec531c86d8f0771778bedc6f50cfe7ecf19db5789aace1990",
}
def lifecycle_plan(current, previous):
    return [("clean", "install", current), ("clean", "list", current),
            ("clean", "uninstall", None), ("clean", "list", None),
            ("upgrade", "install", previous), ("upgrade", "list", previous),
            ("upgrade", "install", current), ("upgrade", "list", current),
            ("upgrade", "uninstall", None), ("upgrade", "list", None)]



def artifact(path, version):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ARTIFACTS[version]:
        raise ValueError("VSIX SHA-256 does not match the pinned " + version + " artifact")
    with zipfile.ZipFile(path) as bundle:
        manifest = json.loads(bundle.read("extension/package.json"))
    identity = manifest.get("publisher", "") + "." + manifest.get("name", "")
    if identity != EXTENSION or manifest.get("version") != version:
        raise ValueError("VSIX package identity or version does not match")
    return {"version": version, "extension_id": identity, "sha256": digest}


def command(cli, workspace, scenario, action, version, artifacts):
    args = [str(cli), "--user-data-dir", str(workspace / scenario / "profile"),
            "--extensions-dir", str(workspace / scenario / "extensions")]
    if action == "install":
        args += ["--install-extension", str(artifacts[version]), "--force"]
    elif action == "uninstall":
        args += ["--uninstall-extension", EXTENSION]
    else:
        args += ["--list-extensions", "--show-versions"]
    return args


def public_command(scenario, action, version):
    args = ["<CODE_CLI>", "--user-data-dir", f"<SHORT_TEMP>/{scenario}/profile",
            "--extensions-dir", f"<SHORT_TEMP>/{scenario}/extensions"]
    if action == "install":
        args += ["--install-extension", f"<VSIX_{version}>", "--force"]
    elif action == "uninstall":
        args += ["--uninstall-extension", EXTENSION]
    else:
        args += ["--list-extensions", "--show-versions"]
    return args


def run_cli(args, env, timeout):
    child = subprocess.Popen(args, env=env, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            stdout, stderr = child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            stdout, stderr = child.communicate()
    return subprocess.CompletedProcess(args, child.returncode, stdout, stderr), timed_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-version", choices=["0.1.44", "0.1.45"], default="0.1.45")
    parser.add_argument("--vsix-current", type=Path, required=True)
    parser.add_argument("--vsix-previous", type=Path, required=True)
    parser.add_argument("--code-cli", type=Path, default=Path(shutil.which("code") or
                        "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("This reproduction is scoped to macOS")
    previous_version = {"0.1.44": "0.1.43", "0.1.45": "0.1.44"}[args.current_version]
    paths = {previous_version: args.vsix_previous.resolve(), args.current_version: args.vsix_current.resolve()}
    steps = lifecycle_plan(args.current_version, previous_version)
    try:
        provenance = {v: artifact(p, v) for v, p in paths.items()}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.error("Artifact validation failed (" + type(error).__name__ + "); no CLI executed")
    plan = [{"scenario": s, "action": a, "expected_version": v,
             "command": public_command(s, a, v)} for s, a, v in steps]
    if not args.execute:
        print(json.dumps({"status": "plan_only", "artifacts": provenance, "plan": plan}, indent=2))
        return 0
    cli = args.code_cli.resolve()
    if not cli.is_file() or not os.access(cli, os.X_OK):
        parser.error("A usable installed VS Code CLI is required")
    stamp = datetime.datetime.now(datetime.timezone.utc)
    output = (args.output or Path("extension-lifecycle-" + stamp.strftime("%Y%m%dT%H%M%SZ"))).resolve()
    if output.exists():
        parser.error("Use a new output directory to preserve earlier results")
    output.mkdir(mode=0o700, parents=True)
    private = output / "private"
    private.mkdir(mode=0o700)
    # The version-only command uses a minimal environment. Lifecycle commands
    # receive a disposable HOME below; no provider or SSH-agent variables pass.
    env = {k: os.environ[k] for k in ("HOME", "LANG", "LC_ALL", "LC_CTYPE") if k in os.environ}
    env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    env["TMPDIR"] = "/tmp"
    report = {
        "checked_at_utc": stamp.isoformat(), "scope": "Published VSIX CLI lifecycle only",
        "platform": {"macos": platform.mac_ver()[0], "architecture": platform.machine(),
                     "runner_python": platform.python_version()},
        "artifacts": provenance, "checks": [], "gui_acceptance": "not_run",
        "raw_logs": "Local private directory; do not publish without review",
    }
    try:
        version, version_timeout = run_cli([str(cli), "--version"], env, 30)
        lines = version.stdout.splitlines()
        if len(lines) >= 3 and re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", lines[0]):
            report["code_version"] = lines[0]
            if re.fullmatch(r"[0-9a-f]{40}", lines[1]):
                report["code_commit"] = lines[1]
            if lines[2] in {"arm64", "x64"}:
                report["code_architecture"] = lines[2]
        with tempfile.TemporaryDirectory(prefix="tvz-", dir="/tmp") as directory:
            workspace = Path(directory)
            isolated_home = workspace / "home"
            isolated_home.mkdir(mode=0o700)
            env["HOME"] = str(isolated_home)
            for index, (scenario, action, expected) in enumerate(steps):
                started = time.monotonic()
                item = dict(plan[index])
                try:
                    result, timed_out = run_cli(command(cli, workspace, scenario, action, expected, paths), env, 180)
                    raw = result.stdout + "\nSTDERR\n" + result.stderr
                    (private / f"{index:02d}-{scenario}-{action}.log").write_text(raw)
                    listed = [line.strip() for line in result.stdout.splitlines()
                              if re.fullmatch(r"[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+@[A-Za-z0-9.+-]+", line.strip())]
                    expected_list = [EXTENSION + "@" + expected] if expected else []
                    ok = not timed_out and result.returncode == 0 and (action != "list" or listed == expected_list)
                    item.update(exit_code=result.returncode, status="passed" if ok else "failed")
                    if timed_out:
                        item.update(status="blocked", reason="CLI timeout")
                    if action == "list":
                        item["tensorviz_versions"] = [line.split("@", 1)[1] for line in listed
                                                     if line.startswith(EXTENSION + "@")]
                        item["other_extension_count"] = sum(not x.startswith(EXTENSION + "@") for x in listed)
                except OSError as error:
                    item.update(exit_code=None, status="blocked", reason=type(error).__name__)
                item["duration_seconds"] = round(time.monotonic() - started, 2)
                report["checks"].append(item)
                if item["status"] != "passed":
                    break
    except (OSError, subprocess.TimeoutExpired) as error:
        report["runner_error"] = type(error).__name__
    report["status"] = "passed" if len(report["checks"]) == len(steps) and all(
        c["status"] == "passed" for c in report["checks"]) else "incomplete_or_failed"
    (output / "install-lifecycle.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "completed_steps": len(report["checks"]),
                      "expected_steps": len(steps)}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

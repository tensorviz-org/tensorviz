#!/usr/bin/env python3
"""Publish an explicitly authorized corpus as real PRs, serially and resumably.

Uses the authenticated gh CLI; never reads or persists its credentials. Each
successful response is checkpointed before the next request. Respect GitHub's
Retry-After and primary reset headers, with bounded secondary-limit retries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def api(endpoint, method="GET", payload=None):
    for attempt in range(7):
        command = ["gh", "api", "--include", "-X", method, endpoint]
        if payload is not None: command += ["--input", "-"]
        result = subprocess.run(command, input=json.dumps(payload) if payload is not None else None, capture_output=True, text=True)
        headers, _, body = result.stdout.replace("\r\n", "\n").partition("\n\n")
        if result.returncode == 0: return json.loads(body)
        values = dict(line.lower().split(": ", 1) for line in headers.splitlines() if ": " in line)
        throttled = "rate limit" in (body + result.stderr).lower() or "retry-after" in values
        if not throttled or attempt == 6: raise RuntimeError(f"GitHub {method} {endpoint}: {body or result.stderr}")
        delay = max(60 * 2**attempt, float(values.get("retry-after", "0")))
        if values.get("x-ratelimit-remaining") == "0": delay = max(delay, float(values.get("x-ratelimit-reset", "0")) - time.time() + 5)
        print(f"GitHub requested backoff; waiting {round(delay)} seconds (attempt {attempt+1}/7)", flush=True)
        time.sleep(delay)
    raise AssertionError("Unreachable")


def publish(repository, state):
    manifest = json.loads((state / "manifest.json").read_text())
    receipt_path = state / "github-prs.jsonl"
    existing = {}
    page = 1
    while True:
        pulls = api(f"repos/{repository}/pulls?state=all&per_page=100&page={page}")
        for pr in pulls: existing[pr["head"]["ref"]] = pr
        if len(pulls) < 100: break
        page += 1
    receipts = {row["id"]: row for row in map(json.loads, receipt_path.read_text().splitlines())} if receipt_path.exists() else {}
    last_write = time.time()
    for item in manifest:
        if item["id"] in receipts: continue
        pr = existing.get(item["branch"])
        if pr is None:
            # At most 450/hour, below the documented general 500/hour ceiling.
            # Measure starts so the API response time does not compound the delay.
            time.sleep(max(0, 8 - (time.time() - last_write)))
            expectation = "No structural graph changes expected." if item["oracle"]["unchanged"] else "The changed computation must be visible or explicitly reported as unsupported; a clean no-change result is a failure."
            body = f"""Graph review stress case **{item['id']}**: {item['family']} ({item['domain']}), width {item['width']}, depth {item['depth']}.

This PR applies `{item['kind']}` to `{item['folder']}/model.py` or its imported child. Both revisions passed a finite PyTorch CPU forward before publication.

{expectation}

Expected target: `{item['oracle']['target']}`. Base: `{item['base']}`. Head: `{item['head']}`.

The test uses TensorViz's production Git source reader, HTTP source projector, and graph comparison. Per-PR evidence will be recorded on the `review-results` branch and tied to these exact GitHub revision SHAs. Unsupported coverage and initial failures are retained explicitly. This fixture is a representative architecture pattern, not a pretrained production model.
"""
            last_write = time.time()
            pr = api(f"repos/{repository}/pulls", "POST", {"title": f"[{item['id']}] {item['kind'].replace('_', ' ')}", "head": item["branch"], "base": "main", "body": body})
        assert pr["head"]["sha"] == item["head"] and pr["base"]["sha"] == item["base"], f"Revision mismatch: {item['id']}"
        receipt = {"id": item["id"], "number": pr["number"], "url": pr["html_url"], "base": pr["base"]["sha"], "head": pr["head"]["sha"], "branch": pr["head"]["ref"]}
        with receipt_path.open("a") as file: file.write(json.dumps(receipt) + "\n"); file.flush()
        receipts[item["id"]] = receipt
        print(f"PUBLISHED {len(receipts)}/1000 {receipt['url']}", flush=True)
    print("COMPLETE: 1000 real GitHub pull requests", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    publish(args.repository, args.state.resolve())

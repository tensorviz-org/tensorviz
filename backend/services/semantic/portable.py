"""Prepare lossless computation evidence with detached, pseudonymous source names.

Never execute/import workspace code. Potentially private argument literals refuse
export instead of being dropped or replaced with a plausible computational value.
This is a payload producer; browser transfer and the graph-file envelope own their
independent transport/file budgets.
"""
from __future__ import annotations

import hashlib
import json

from .contracts import CaptureSnapshot, PortableCapture


def portable_capture(payload):
    capture = CaptureSnapshot.model_validate(payload)
    result = capture.model_dump(mode="json")
    source = result["source"]
    source_id = "portable:" + hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    files = {file["path"]: "tensorviz.toml" if file["role"] == "config" else f"file-{index}.py"
             for index, file in enumerate(source["files"])}
    source.update(id=source_id, workspaceSessionId="portable", workspaceId="portable", modelId="portable",
                  generation=0, modelFile=files[source["modelFile"]])
    for file in source["files"]:
        file.update(path=files[file["path"]], documentVersion=None)
    for binding in result["sourceBindings"]:
        binding["snapshotId"] = source_id
        for anchor in binding["anchors"]:
            anchor["file"] = files[anchor["file"]]
    return PortableCapture(version=1, sourceNames="anonymized", capture=result)

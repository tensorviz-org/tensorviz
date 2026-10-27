"""Saving and restoring. Weights only — an optimizer state doubles the file."""
import json
from pathlib import Path

import torch


def save_checkpoint(model, path: str = "runs/tiny_lm.pt", meta=None):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, destination)
    return str(destination)


def load_checkpoint(model, path: str = "runs/tiny_lm.pt"):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return payload.get("meta", {})


def write_manifest(path: str, values: dict):
    Path(path).write_text(json.dumps(values, indent=2), encoding="utf-8")

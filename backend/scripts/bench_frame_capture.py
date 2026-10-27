"""Measure what per-op source-frame capture costs a real trace (task.md M4).

Per-op frame capture is the expensive half of M4: ``_Provenance._source_frame`` walks
the Python stack once for **every recorded aten op**, and a gpt2 forward records
hundreds of them (1621 raw, ~885 after the noise filter). The milestone's rule is that
the capture ships on by default only if it costs less than ~25%; above that it is
flag-gated and the extension turns it on per request.

This script is how that number is decided, and it is deliberately the WORST case: the
frame root is the directory the model's own class lives in, so every op finds a frame
and the walk never short-circuits early on a miss.

    KMP_DUPLICATE_LIB_OK=TRUE <python> backend/scripts/bench_frame_capture.py

Models are skipped, loudly, when their library is not installed — a benchmark that
silently measures three models instead of five is worse than one that says so.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from services.ingest.introspect import class_source  # noqa: E402
from services.ingest.trace import trace_model  # noqa: E402

#: Above this, the milestone says capture must be flag-gated rather than default-on.
BUDGET_PERCENT = 25.0

#: The budget is about big graphs — gpt2 emits ~1621 raw ops, and the question it
#: asks is whether a per-op stack walk is affordable there. A nine-op toy answers a
#: different question: its trace costs ~50 us in total, so a fixed 1.5 us per op is
#: a fifth of it while being nothing at all. Both rows are printed; only rows in the
#: regime the budget is about decide the verdict, and the per-op microseconds below
#: are the scale-free number that lets a reader check that judgement.
BUDGET_MIN_OPS = 50


def _fixture() -> Tuple[Any, Tuple[Any, ...]]:
    """The IDE fixture net: the small end, where fixed costs dominate."""
    from services.ingest.repo_scan import import_with_stubs

    root = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "ide_repo")
    module, _ = import_with_stubs("models.resid", [os.path.abspath(root)])
    return module.Net(), (torch.randn(2, 8),)


def _torchvision(arch: str) -> Callable[[], Tuple[Any, Tuple[Any, ...]]]:
    def build() -> Tuple[Any, Tuple[Any, ...]]:
        import torchvision.models as tv

        return getattr(tv, arch)(weights=None), (torch.randn(1, 3, 224, 224),)

    return build


def _transformer() -> Tuple[Any, Tuple[Any, ...]]:
    """A 12-layer encoder built from torch.nn only.

    Stands in for gpt2 in the one dimension this benchmark cares about — op count in
    the high hundreds — on a machine without transformers installed. It is not a
    substitute for the gpt2 row when transformers IS available: a real model runs its
    own Python between ops, and that is where the stack walk is longest.
    """
    import torch.nn as nn

    layer = nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128, batch_first=True
    )
    return nn.TransformerEncoder(layer, num_layers=12), (torch.randn(1, 16, 64),)


def _gpt2() -> Tuple[Any, Tuple[Any, ...]]:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained("gpt2")
    return AutoModel.from_config(config), (torch.randint(0, 1000, (1, 16)),)


MODELS: Sequence[Tuple[str, Callable[[], Tuple[Any, Tuple[Any, ...]]]]] = (
    ("ide_repo:Net", _fixture),
    ("torchvision:resnet18", _torchvision("resnet18")),
    ("torchvision:mobilenet_v2", _torchvision("mobilenet_v2")),
    ("torch.nn:transformer-12L", _transformer),
    ("transformers:gpt2", _gpt2),
)


def _roots_for(model: Any) -> Tuple[str, ...]:
    """The directory the model's class was defined in — where its frames will be."""
    file, _line = class_source(type(model))
    return (os.path.dirname(os.path.abspath(file)),) if file else ()


#: A sample shorter than this is timer noise, not a measurement. The fixture net
#: traces in ~0.1 ms, where scheduler jitter alone moved the reported overhead
#: between 3% and 19% run to run — enough to fail a 25% budget by accident.
_MIN_SAMPLE_MS = 5.0


def _batch_size(model: Any, args: Tuple[Any, ...], roots: Sequence[str]) -> int:
    """How many traces to time together so one sample is worth reading."""
    started = time.perf_counter()
    trace_model(model, args, {}, tier="real", frame_roots=roots)
    one = (time.perf_counter() - started) * 1000
    return max(1, min(2000, int(_MIN_SAMPLE_MS / one) + 1)) if one > 0 else 1


def _time(
    model: Any, args: Tuple[Any, ...], roots: Sequence[str], repeats: int, batch: int
) -> List[float]:
    samples: List[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(batch):
            result = trace_model(model, args, {}, tier="real", frame_roots=roots)
        samples.append((time.perf_counter() - started) * 1000 / batch)
        if result.error:
            raise RuntimeError(f"trace failed: {result.error}")
    return samples


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    arguments = parser.parse_args(argv)

    print(
        f"{'model':<28}{'ops':>7}{'frames':>8}{'off ms':>10}{'on ms':>10}"
        f"{'overhead':>10}{'us/op':>8}{'batch':>7}"
    )
    overheads: Dict[str, float] = {}
    counted: Dict[str, float] = {}
    skipped: List[str] = []
    for name, build in MODELS:
        try:
            model, args = build()
        except Exception as exc:
            skipped.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        model.eval()
        roots = _roots_for(model)
        if not roots:
            skipped.append(f"{name}: no source file for {type(model).__name__}")
            continue

        batch = _batch_size(model, args, roots)
        _time(model, args, (), arguments.warmup, batch)
        off = _time(model, args, (), arguments.repeats, batch)
        on = _time(model, args, roots, arguments.repeats, batch)
        traced = trace_model(model, args, {}, tier="real", frame_roots=roots)
        median_off = statistics.median(off)
        median_on = statistics.median(on)
        overhead = (median_on - median_off) / median_off * 100 if median_off else 0.0
        ops = len(traced.ops)
        per_op = (median_on - median_off) * 1000 / ops if ops else 0.0
        overheads[name] = overhead
        if ops >= BUDGET_MIN_OPS:
            counted[name] = overhead
        print(
            f"{name:<28}{ops:>7}{traced.frames_captured:>8}"
            f"{median_off:>10.2f}{median_on:>10.2f}{overhead:>9.1f}%{per_op:>8.2f}{batch:>7}"
        )

    for line in skipped:
        print(f"skipped {line}")
    if not overheads:
        print("no model could be benchmarked; nothing was measured")
        return 1
    if not counted:
        print(f"no model reached {BUDGET_MIN_OPS} ops; the budget was not applied to anything")
        return 1

    peak_name = max(counted, key=lambda key: counted[key])
    print(
        f"\nworst of the {len(counted)} model(s) with >= {BUDGET_MIN_OPS} ops: "
        f"{peak_name} at {counted[peak_name]:.1f}% (budget {BUDGET_PERCENT:.0f}%)"
    )
    if counted[peak_name] > BUDGET_PERCENT:
        print(
            "over budget: run the worker with --no-frames by default and turn capture on "
            "per request (services/ingest/worker.py)"
        )
        return 2
    print("within budget: frame capture stays on by default")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluation summaries must not turn missing/failed evidence into an adapter pass."""
from types import SimpleNamespace

from scripts.benchmark_semantic_adapters import semantic_edges, summarize


def sample(adapter, **changes):
    return dict(case="fixture", adapter=adapter, **dict(status="passed", measureMs=10,
        measurePeakRssBytes=1024, inputDigest="input", stateDigest="state", outputDigest="output") | changes)


def test_unavailable_export_is_kept_separate_from_the_successful_runtime_baseline():
    rows = summarize([sample("raw"), sample("runtime", measureMs=20),
                      sample("export_non_strict", status="failed", measureMs=1)])[0]["adapters"]
    assert rows["runtime"]["matchesRawIdentity"] and rows["runtime"]["timeToRawRatio"] == 2
    assert rows["export_non_strict"] == dict(passed=0, samples=1)


def test_missing_or_inconsistent_state_and_output_never_pass_cross_process_identity():
    for key in ("inputDigest", "stateDigest", "outputDigest"):
        for bad in (None, "different"):
            rows = summarize([sample("raw"), sample("runtime", **{key: bad})])[0]["adapters"]
            assert not rows["runtime"]["matchesRawIdentity"]
        rows = summarize([sample("raw", **{key: None}), sample("runtime", **{key: None})])[0]["adapters"]
        assert not rows["runtime"]["matchesRawIdentity"]
    rows = summarize([sample("raw"), sample("raw", stateDigest="other"), sample("runtime")])[0]["adapters"]
    assert not rows["runtime"]["matchesRawIdentity"]


def test_failed_samples_are_visible_but_cannot_improve_the_timing_median():
    rows = summarize([sample("raw"), sample("runtime", measureMs=20),
        sample("runtime", measureMs=40), sample("runtime", status="timeout", measureMs=0)])[0]["adapters"]
    assert rows["runtime"]["samples"] == 3 and rows["runtime"]["passed"] == 2
    assert rows["runtime"]["medianMs"] == 30


def test_collapse_may_change_display_ports_but_not_original_argument_uses():
    fields = dict(id="edge", kind="argument", valueId="value", sourcePortId="out", targetPortId="args[1]")
    full = SimpleNamespace(edges=[SimpleNamespace(**fields, displaySourcePortId="out")])
    grouped = SimpleNamespace(edges=[SimpleNamespace(**fields, displaySourcePortId="group")])
    assert semantic_edges(full) == semantic_edges(grouped)
    grouped.edges[0].targetPortId = "args[0]"
    assert semantic_edges(full) != semantic_edges(grouped)

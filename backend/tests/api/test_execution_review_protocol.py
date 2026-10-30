from copy import deepcopy

from services.ide.graph_setup_protocol import action, execution_review, opening


def state():
    return dict(id="review1", phase="review", kind="capture", device="cuda:0", configuredDevice="cuda:0", detail="Not started",
                inputs="Saved inputs", mode="eval", gradients=False, interpreter="/env/bin/python",
                stages=[dict(label="Capture", calls=1)], resources=dict(sampledAt=1234, host="gpu-host", remote=True,
                cpu="CPU", cores=16, cpuPercent=80, memoryFree=1024, memoryTotal=8192,
                devices=[dict(id="cuda:0", name="GPU", utilization=100, memoryFree=0, memoryTotal=8192)], detail="Observed only"))


def test_execution_readings_and_unknowns_survive_the_broker_wire():
    value = state()
    assert execution_review(value)
    value["resources"]["devices"][0]["utilization"] = None
    assert execution_review(value)
    assert opening(dict(id="opening", target={}, phase="awaiting_execution", detail="Review", hasCapture=False,
                        execution=value), lambda _: True)
    for invalid in (float("nan"), -1, True):
        changed = deepcopy(value)
        changed["resources"]["cpuPercent"] = invalid
        assert not execution_review(changed)


def test_only_bounded_device_actions_can_cross_the_broker():
    value = dict(kind="execution", reviewId="review1", action="run", device="cuda:0")
    assert action(value)
    assert not action({**value, "device": "cuda:0;sh"})
    assert not action({**value, "action": "auto_run"})
    assert not action({**value, "command": "python model.py"})


def test_timing_requests_and_latency_reviews_no_longer_cross_the_broker():
    timing = dict(warmup=2, iterations=7, profileIterations=4)
    configure = dict(kind="execution", reviewId="review1", action="configure", device="cpu", timing=timing)
    assert not action(configure)
    assert not action({**configure, "timing": None})
    assert not execution_review({**state(), "timing": timing})
    assert not execution_review({**state(), "kind": "latency"})


def test_saved_profile_navigation_crosses_the_broker_without_commands():
    assert execution_review({**state(), "profileName": "run.v2"})
    edit = dict(kind="edit_execution", reviewId="review1", field=["inputs", "args", 0, "shape"])
    assert action(edit)
    for field in ([], ["profile"], ["inputs", -1], ["inputs", 64], ["mode", "other"]):
        assert not action({**edit, "field": field})
    assert not action({**edit, "path": "/other/tensorviz.toml"})


def test_reused_result_label_and_run_fresh_cross_the_broker_only_on_finished_reviews():
    reused = {**state(), "phase": "succeeded", "reused": {"capturedAt": 1761400000000}}
    assert execution_review(reused)
    assert execution_review({**state(), "fresh": True})
    for invalid in ({**reused, "phase": "running"}, {**reused, "reused": {"capturedAt": "14:02"}},
                    {**reused, "reused": {"capturedAt": -1}}, {**reused, "reused": {"capturedAt": 1, "attemptId": "a"}},
                    {**reused, "fresh": True}, {**state(), "fresh": False}):
        assert not execution_review(invalid)

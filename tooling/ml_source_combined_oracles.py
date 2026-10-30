"""Actual CPU controls for changed tensor reads and conflicting shared constructors.

The original width oracle is preserved. These cases exercise additional limits,
including a successful candidate which violates the unchanged-constructor-state
assumption and must therefore remain unknown to the conditional analyzer.
"""
import argparse
from fixture_json import dumps
from pathlib import Path
from ml_source_change_oracle import build, SOURCES, CONFIG


def variants():
    reads = dict(SOURCES)
    reads["model.py"] = reads["model.py"].replace(
        "hidden.reshape(x.shape[0], HEADS, WIDTH // HEADS)", "hidden.reshape(hidden.shape[0], hidden.shape[1])")
    shared = {CONFIG + ".py": SOURCES[CONFIG + ".py"], "model.py": f"""import torch
from {CONFIG} import WIDTH
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, WIDTH, bias=False)
        self.tied = self.up
        self.other = torch.nn.Linear(8, 24 - WIDTH, bias=False)
        self.other.weight = self.up.weight
    def forward(self, x):
        return self.up(x) + self.other(x)
"""}
    return dict(version=1, cases=[dict(name="changed_tensor_read", **build(reads, ("passed",) * 3)),
        dict(name="conflicting_shared_constructors", **build(shared, ("passed",) * 3))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    result = variants()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(dumps(result) + "\n")
    print("Wrote two real captures, four source-query pages and six independent CPU forwards")

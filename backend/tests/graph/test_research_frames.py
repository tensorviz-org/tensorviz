"""A function body as a dataflow graph (step 5.3).

The card for a trainer says "calls .backward() and an optimizer .step()" — true,
and the whole of what the project graph knew about the one place training happens.
`inputs -> logits -> loss -> backward -> step` is inside, and a function drawn as
one box hides it.

Variables are the edges. That is not a guess about intent: a step that reads a name
an earlier step wrote is downstream of it, and Python's scoping says which write a
read refers to.
"""



from research.frames import MAX_STEPS, steps_of  # noqa: E402

LOOP = '''
import torch
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader
        self.optimizer = torch.optim.AdamW(model.parameters())

    def train(self, epochs=1):
        for epoch in range(epochs):
            for inputs, targets in self.loader:
                logits = self.model(inputs)
                loss = F.cross_entropy(logits, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("loss", loss.item())
'''


def labels(steps):
    return [step.label for step in steps]


def test_the_loop_is_read_out_of_a_class_not_its_init():
    """A trainer is usually a CLASS, and a class has no statements worth a frame:
    `__init__` assigns fields and the loop is elsewhere. The method that calls
    `.backward()` is the loop, which is evidence rather than a naming guess."""
    steps = steps_of(LOOP, "Trainer")
    assert "F.cross_entropy" in labels(steps)
    # __init__'s assignments are not the frame.
    assert "torch.optim.AdamW" not in labels(steps)


def test_a_variable_makes_the_edge():
    steps = steps_of(LOOP, "Trainer")
    by_label = {step.label: step for step in steps}
    logits = by_label["self.model"]
    loss = by_label["F.cross_entropy"]
    backward = by_label["loss.backward"]
    assert logits.writes == "logits"
    assert loss.reads == {logits.id}
    assert backward.reads == {loss.id}


def test_the_two_steps_that_make_it_training_are_marked():
    """`backward` and the optimizer step are what a reader looks for first."""
    roles = {step.label: step.role for step in steps_of(LOOP, "Trainer")}
    assert roles["loss.backward"] == "gradient"
    assert roles["self.optimizer.step"] == "optimizer"


def test_logging_is_not_a_step():
    """A `print` per iteration doubles the frame and says nothing about what flows
    where. It is something the code does ABOUT the pipeline, not part of it."""
    assert "print" not in labels(steps_of(LOOP, "Trainer"))


def test_a_loop_body_is_flattened_rather_than_drawn_as_control_flow():
    """A reader looking at a training loop wants the pipeline. Drawing the back edge
    of every `for` would be a diagram of Python rather than of the model."""
    steps = steps_of(LOOP, "Trainer")
    # Both `for` headers are gone; everything inside them is present and in order.
    assert labels(steps).index("self.model") < labels(steps).index("loss.backward")


def test_a_plain_function_works_too():
    source = '''
def evaluate(model, loader):
    total = 0
    for batch in loader:
        out = model(batch)
        score = metric(out)
        total = total + score
    return total
'''
    steps = steps_of(source, "evaluate")
    by_label = {step.label: step for step in steps}
    assert by_label["metric"].reads == {by_label["model"].id}


def test_a_name_reassigned_chains_through_its_own_history():
    """Resolved against the most recent EARLIER write, which is what Python does.
    Against the first write instead, a running total would fan out from its
    initialiser and the sequence would be lost."""
    source = '''
def run(loader):
    total = 0
    for batch in loader:
        total = total + step(batch)
        total = clip(total)
    return total
'''
    steps = steps_of(source, "run")
    assert [step.label for step in steps][:3] == ["total", "total", "clip"]
    # `clip` reads the addition, not the `total = 0` that came before it.
    assert steps[2].reads == {steps[1].id}


def test_a_body_that_will_not_parse_yields_nothing_rather_than_raising():
    """One bad function must not blank a project."""
    assert steps_of("def broken(\n", "broken") == []


def test_a_very_long_body_is_capped():
    """A 400-line `main` is a file to open, not a frame to read."""
    body = "\n".join(f"    x{i} = call{i}()" for i in range(MAX_STEPS + 20))
    assert len(steps_of(f"def main():\n{body}\n", "main")) == MAX_STEPS


def test_a_name_the_frame_does_not_define_makes_no_edge():
    """`self` is read by every step in a method; an edge for it joins everything to
    everything. So are builtins, and so is anything defined outside the body."""
    steps = steps_of(LOOP, "Trainer")
    for step in steps:
        assert all(other in {s.id for s in steps} for other in step.reads)

"""Services behind ``/api/ide/*``: the code <-> graph projection (task.md §5).

Kept out of ``services/ingest`` on purpose. That package imports torch and
pydantic at module scope and executes user code to trace a model; everything
here is static, reads source only through ``ast``, and never imports or runs a
line of the user's program (task.md §8.5).
"""

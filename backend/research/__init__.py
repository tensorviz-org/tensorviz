"""The project-scale graph: a whole ML project read from source, never run.

Where `services/ide/` projects one `nn.Module` class into a layer graph, this
package projects a *project*: which experiments exist, what data feeds what,
which model trains where, and how every node points back at the code that
declares it. See `docs/project-graph-plan.md`.

Every module here is pure. Nothing in this package opens a file, touches a
database or imports the user's code — the extension supplies source text and
receives data back. `tests/graph/test_research_purity.py` enforces that rather than
leaving it to convention.
"""

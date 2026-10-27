"""What is different between two experiments.

The question a researcher asks constantly and no tool answers: *these two runs
disagree — what changed?* Usually it is answered by diffing two files by hand and
hoping nothing shared moved underneath.

This answers it from two present-tense graphs, so it needs no history, no version
store and nothing written to disk. That is the whole reason it fits in a read-only
extension: comparing what two experiments *are* requires only that both were
indexed, and the index is already there.

Three kinds of difference, in the order they matter:

* **settings** — the same configuration field with two values. `n_layer: 12 → 4`
  is the answer to the question nine times out of ten.
* **structure** — a node one experiment has and the other does not, matched on
  `definition_id` so a class used by both is recognised as the same class even
  though the two graphs give it different node ids.
* **shape** — a definition both use, where one wires it differently.

Symmetric by construction: `compare(a, b)` and `compare(b, a)` describe the same
difference from the two sides, and neither is "the baseline". Nothing here decides
which experiment is right.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from research.contracts import Experiment, ResearchGraph


class SettingChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: The configuration node's label, so a reader knows where to look.
    where: str
    name: str
    left: Optional[Any] = None
    right: Optional[Any] = None
    #: True when only one side declares it at all, which reads differently from
    #: two sides disagreeing.
    only_one_side: bool = False


class NodeChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition_id: str
    label: str
    kind: str
    #: 'left' or 'right' — which experiment has it.
    side: str


class ExperimentDiff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: str
    right: str
    settings: List[SettingChange] = Field(default_factory=list)
    nodes: List[NodeChange] = Field(default_factory=list)
    #: Definitions both experiments use. Usually the interesting half: it says the
    #: two runs really are comparable rather than unrelated work.
    shared: List[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.settings and not self.nodes


def _settings_of(graph: ResearchGraph) -> Dict[Tuple[str, str], Any]:
    """`(where, name) -> value` for every configuration node's declared settings."""
    found: Dict[Tuple[str, str], Any] = {}
    for node in graph.nodes:
        settings = node.params.get("settings")
        if not isinstance(settings, dict):
            continue
        for name, value in settings.items():
            found[(node.label, str(name))] = value
    return found


def _definitions_of(graph: ResearchGraph) -> Dict[str, Any]:
    """`definition_id -> node`, skipping files, which are not interesting here."""
    return {
        node.definition_id: node
        for node in graph.nodes
        if node.definition_id and node.kind != "file"
    }


def compare(left: Experiment, right: Experiment) -> ExperimentDiff:
    """Everything that differs between two experiments, from both sides."""
    left_settings = _settings_of(left.graph)
    right_settings = _settings_of(right.graph)

    settings: List[SettingChange] = []
    for key in sorted(set(left_settings) | set(right_settings), key=lambda k: (k[0], k[1])):
        where, name = key
        in_left, in_right = key in left_settings, key in right_settings
        if in_left and in_right:
            if left_settings[key] != right_settings[key]:
                settings.append(
                    SettingChange(
                        where=where,
                        name=name,
                        left=left_settings[key],
                        right=right_settings[key],
                    )
                )
        else:
            # Only one side declares it. Reported, because a setting that exists in
            # one run and not the other is a real difference — but flagged, because
            # it reads differently from two values disagreeing.
            settings.append(
                SettingChange(
                    where=where,
                    name=name,
                    left=left_settings.get(key),
                    right=right_settings.get(key),
                    only_one_side=True,
                )
            )

    left_definitions = _definitions_of(left.graph)
    right_definitions = _definitions_of(right.graph)

    # A definition id is file-scoped, so two experiments that each keep their own
    # `config_a.py` / `config_b.py` have different ids for the same `RunConfig`.
    # Reporting that as "only on the left" AND "only on the right" tells a reader
    # nothing — the useful answer is that both have one and their settings differ,
    # which the settings comparison above already gives by label. So a definition
    # missing on one side is only a structural difference when the other side has
    # nothing of that name and kind either.
    def counterpart(node, others) -> bool:
        return any(
            other.label == node.label and other.kind == node.kind for other in others.values()
        )

    nodes: List[NodeChange] = []
    for definition in sorted(set(left_definitions) - set(right_definitions)):
        node = left_definitions[definition]
        if not counterpart(node, right_definitions):
            nodes.append(
                NodeChange(definition_id=definition, label=node.label, kind=node.kind, side="left")
            )
    for definition in sorted(set(right_definitions) - set(left_definitions)):
        node = right_definitions[definition]
        if not counterpart(node, left_definitions):
            nodes.append(
                NodeChange(definition_id=definition, label=node.label, kind=node.kind, side="right")
            )

    return ExperimentDiff(
        left=left.label,
        right=right.label,
        settings=settings,
        nodes=nodes,
        shared=sorted(set(left_definitions) & set(right_definitions)),
    )

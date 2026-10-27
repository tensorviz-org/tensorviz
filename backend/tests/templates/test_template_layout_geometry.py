"""Layout guard: an instantiated template must be *readable* on the canvas.

Every root node renders as a fixed-size card (tldraw LayerShapeUtil /
InputShapeUtil / OutputShapeUtil all default to w=200, h=120), so the positions
a template ships decide whether the user sees a legible left-to-right chain or a
pile of cards stacked on top of each other. Two failure modes used to ship:

* ``TemplateService._ensure_io_nodes`` synthesized Input and Output at the same
  ``{"x": 0, "y": 0}``, so Input was 100% hidden underneath Output; and
* the starter grid stepped x by 170px for 200px-wide cards, guaranteeing a 30px
  overlap between every adjacent pair.

Both are invisible to shape/validation tests — this file is the only thing that
can fail when the geometry regresses.
"""



import pytest


from services.template_service import TemplateService  # noqa: E402

# Rendered card footprint (tldraw shape-util default props).
CARD_W = 200
CARD_H = 120
CARD_AREA = CARD_W * CARD_H
# A little overlap is tolerable (cards can nudge); half a card hidden is not.
MAX_OVERLAP_FRACTION = 0.25

_SVC = TemplateService()
_TEMPLATE_IDS = sorted(
    tid
    for cat in (
        _SVC.starter_templates,
        _SVC.transformer_templates,
        _SVC.cv_templates,
        _SVC.nlp_templates,
        _SVC.generative_templates,
    )
    for tid in cat.get_all_templates()
)


def _overlap_area(a, b) -> float:
    dx = min(a[0] + CARD_W, b[0] + CARD_W) - max(a[0], b[0])
    dy = min(a[1] + CARD_H, b[1] + CARD_H) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _offences(template_id):
    """Return (identical-coordinate pairs, over-tolerance overlapping pairs)."""
    graph = _SVC.instantiate_template(template_id, {})
    placed = [(n.id, (n.position.get("x", 0.0), n.position.get("y", 0.0)))
              for n in graph.root_nodes]
    identical, overlapping = [], []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            (id_a, pos_a), (id_b, pos_b) = placed[i], placed[j]
            if pos_a == pos_b:
                identical.append((id_a, id_b, pos_a))
                continue
            frac = _overlap_area(pos_a, pos_b) / CARD_AREA
            if frac > MAX_OVERLAP_FRACTION:
                overlapping.append((id_a, id_b, round(frac, 3)))
    return identical, overlapping


@pytest.mark.parametrize("template_id", _TEMPLATE_IDS)
def test_template_nodes_do_not_collide(template_id):
    identical, overlapping = _offences(template_id)
    assert not identical, (
        f"{template_id}: nodes share identical coordinates (one is completely "
        f"hidden under the other): {identical}"
    )
    assert not overlapping, (
        f"{template_id}: node cards overlap by more than "
        f"{int(MAX_OVERLAP_FRACTION * 100)}% of a {CARD_W}x{CARD_H} card: "
        f"{overlapping}"
    )


def test_no_template_ships_colliding_layout():
    """Aggregate view so a regression names every offending template at once."""
    offenders = {}
    for template_id in _TEMPLATE_IDS:
        identical, overlapping = _offences(template_id)
        if identical or overlapping:
            offenders[template_id] = {"identical": identical, "overlapping": overlapping}
    assert not offenders, (
        f"{len(offenders)}/{len(_TEMPLATE_IDS)} templates ship colliding node "
        f"positions: {sorted(offenders)}\ndetail: {offenders}"
    )

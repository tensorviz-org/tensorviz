"""Tests for complex model roadmap metadata."""



from services.model_roadmap import get_complex_model_roadmap  # noqa: E402


def test_complex_model_roadmap_has_ranked_packs():
    roadmap = get_complex_model_roadmap()
    packs = roadmap.get("packs", [])
    assert isinstance(packs, list)
    assert len(packs) >= 5
    priorities = [pack.get("priority") for pack in packs]
    assert all(isinstance(p, int) for p in priorities)
    assert priorities == sorted(priorities)

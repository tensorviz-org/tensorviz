"""The catalogue offers exactly the cards that work — no more, no fewer.

Two failure modes, and this suite pins both. Offering a card that cannot be built lets a
user place a template and then never export or run it. Filtering with the WRONG oracle
silently empties the gallery, which is the worse of the two and has now happened three
times: once reading ops out of hierarchical graphs (dropped thirteen working templates),
once asking `spec_for` for ops the generator ships its own implementations of (dropped
F5-TTS), and once filtering module blocks against a neighbouring catalogue (dropped all
fifteen, including three that instantiate fine).
"""


import server_v2  # noqa: E402


def _offered():
    templates, popular, modules = {}, {}, {}
    for category in server_v2.template_service.get_all_categories():
        cid = category.get("id") or category.get("name")
        body = server_v2.get_templates_by_category(cid)
        templates.update(body["templates"])
        popular.update(body["popular_models"])
        modules.update(body["module_templates"])
    return templates, popular, modules


def test_f5_tts_is_offered_under_both_ids():
    """F5-TTS is the flagship modularity example and must stay in the catalogue.

    Its 290 nodes factor into `DiTBlock x22` via `nn.ModuleList` — the case
    `server_v2.generate_code` names in its own comment as the reason modular codegen
    exists. It was removed once, on the theory that `sinusoidal_pos_embedding` and `grn`
    "carry a name and a colour and no dimensions". They do not: `code_generation` ships
    shape-preserving nn.Module implementations for both (`_CUSTOM_OP_CLASS`), written for
    these very blocks. Asking the op registry instead of the generator was the mistake.
    """
    templates, popular, _ = _offered()
    assert "f5_tts" in templates
    assert "f5tts" in popular


def test_the_catalogue_is_not_quietly_emptied():
    templates, popular, _ = _offered()
    assert len(templates) >= 50, len(templates)
    assert len(popular) >= 15, len(popular)
    for expected in ("resnet", "vgg", "gan", "vae", "lstm_classifier", "f5_tts"):
        assert expected in templates, expected


def test_every_offered_template_passes_the_generator_s_own_validation():
    templates, popular, _ = _offered()
    bad = [tid for tid in templates if not server_v2._buildable_template(tid)]
    bad += [mid for mid in popular if not server_v2._buildable_popular(mid)]
    assert not bad, f"offered but not buildable: {bad}"


def test_every_advertised_module_block_can_be_instantiated():
    """A card that 404s when clicked is worse than no card.

    Twelve advertised ids (`resnet_block`, `lstm_cell`, `attention_mechanism`, ...) resolve
    to nothing. The oracle here is `get_module_template_by_id`, which is what the
    instantiate route itself calls — filtering against the canonical registry, a different
    catalogue behind a different endpoint, dropped every card including the working ones.
    """
    _, _, modules = _offered()
    dead = [
        mid
        for mid in modules
        if server_v2.template_service.get_module_template_by_id(mid) is None
    ]
    assert not dead, f"advertised but not instantiable: {dead}"
    assert modules, "every module block was filtered away — that is a broken filter"


def test_the_known_dead_module_ids_stay_out():
    _, _, modules = _offered()
    for phantom in ("resnet_block", "lstm_cell", "attention_mechanism", "generator_block"):
        assert phantom not in modules, phantom

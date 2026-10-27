

from services.template_staging_service import TemplateAutoGenService, TemplateStagingStore


def test_staging_store_persists_and_reloads(tmp_path):
    registry = tmp_path / "template-staging.json"
    store = TemplateStagingStore(registry_path=str(registry))
    autogen = TemplateAutoGenService(store=store)

    staging_ids = autogen.generate_from_gap(
        gap_id="gap_tts_vocoder",
        missing_templates=["neural_vocoder_block"],
        required_actions=["mel_spectrogram_shape_validation"],
        family="tts",
    )
    assert len(staging_ids) == 1

    reloaded = TemplateStagingStore(registry_path=str(registry))
    rows = reloaded.list_all()
    assert len(rows) == 1
    assert rows[0].template_id == "neural_vocoder_block"
    assert rows[0].source_gap_id == "gap_tts_vocoder"


def test_staging_store_dedupes_gap_template_by_default():
    store = TemplateStagingStore()
    autogen = TemplateAutoGenService(store=store)

    first = autogen.generate_from_gap(
        gap_id="gap_resnet_stage",
        missing_templates=["resnet_stage_block"],
        required_actions=["residual_connection"],
        family="resnet",
    )
    second = autogen.generate_from_gap(
        gap_id="gap_resnet_stage",
        missing_templates=["resnet_stage_block"],
        required_actions=["residual_connection"],
        family="resnet",
    )

    assert first == second
    assert len(store.list_all()) == 1


def test_staging_store_can_allow_duplicates():
    store = TemplateStagingStore()
    autogen = TemplateAutoGenService(store=store)

    first = autogen.generate_from_gap(
        gap_id="gap_unet_decoder",
        missing_templates=["unet_decoder_stage"],
        required_actions=["skip_connection_concat"],
        family="unet",
        allow_duplicates=True,
    )
    second = autogen.generate_from_gap(
        gap_id="gap_unet_decoder",
        missing_templates=["unet_decoder_stage"],
        required_actions=["skip_connection_concat"],
        family="unet",
        allow_duplicates=True,
    )

    assert first != second
    assert len(store.list_all()) == 2

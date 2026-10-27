"""Random-weight corpus factories; copied into an isolated model workspace only."""
import torch
from whisper.model import ModelDimensions, Whisper


def _random_model(dimensions):
    model = Whisper(dimensions)
    # Whisper normally loads this torch.empty parameter from a checkpoint. With
    # deterministic PyTorch enabled it contains NaNs until explicitly initialized.
    # Use the recipe's seeded RNG; never alter the original model implementation.
    torch.nn.init.normal_(model.decoder.positional_embedding, std=0.02)
    return model


def make_model():
    return _random_model(ModelDimensions(
        n_mels=80, n_audio_ctx=16, n_audio_state=32, n_audio_head=4,
        n_audio_layer=2, n_vocab=128, n_text_ctx=16, n_text_state=32,
        n_text_head=4, n_text_layer=2,
    ))


def make_tiny():
    return _random_model(ModelDimensions(
        n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6,
        n_audio_layer=4, n_vocab=51865, n_text_ctx=448, n_text_state=384,
        n_text_head=6, n_text_layer=4,
    ))

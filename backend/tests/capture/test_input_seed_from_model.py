"""The input search must be able to trace the simplest model anyone writes.

The candidate pool holds images and integer token tensors and no plain 2-D float
activation, so an MLP could never be handed `randn(1, N)`. Every candidate failed,
the search fell through to the 1x3x224x224 image, and the user was told
"mat1 and mat2 shapes cannot be multiplied (672x224 and 16x16)" about a model that
is perfectly correct. These pin the seed that reads the answer off the model.
"""


import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from services.ingest import inputs as input_search  # noqa: E402


def test_a_linear_first_model_is_given_its_own_width():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(16, 16)
            self.l2 = nn.Linear(16, 4)

        def forward(self, x):
            return self.l2(self.l1(x))

    seed = input_search.seed_from_model(MLP())
    assert seed is not None
    assert tuple(seed().shape) == (1, 16)


def test_the_seed_is_tried_before_the_image_pool():
    # Order is the whole point: the 224x224 image is candidate 1 today, and it is
    # what produced the mat1/mat2 error on every MLP.
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x):
            return self.fc(x)

    first = input_search.candidates_for(MLP(), 1)[0]
    assert tuple(first[0]().shape) == (1, 8)


def test_an_embedding_first_model_gets_integer_tokens():
    # A float tensor into nn.Embedding raises; the index dtype is the whole point.
    class Tokens(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(500, 32)

        def forward(self, x):
            return self.emb(x)

    seed = input_search.seed_from_model(Tokens())
    assert seed is not None
    produced = seed()
    assert produced.dtype in (torch.int64, torch.int32)
    assert int(produced.max()) < 500


def test_a_conv_first_model_gets_its_channel_count():
    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(7, 4, 3)

        def forward(self, x):
            return self.conv(x)

    seed = input_search.seed_from_model(Vision())
    assert seed is not None
    assert tuple(seed().shape)[:2] == (1, 7)


def test_a_model_whose_first_layer_says_nothing_still_searches():
    # No regression for the repositories the search was built for: when the first
    # layer carries no width, the pool must be returned exactly as before.
    class Opaque(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(x)

    model = Opaque()
    assert input_search.seed_from_model(model) is None
    assert input_search.candidates_for(model, 1) == input_search.candidates(1)


def test_a_multi_argument_forward_is_left_to_the_search():
    # The seed describes the primary activation only; a half-known tuple first
    # would just cost a failed forward.
    class TwoArg(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(8, 8)

        def forward(self, x, t):
            return self.l1(x) * t

    model = TwoArg()
    assert input_search.candidates_for(model, 2) == input_search.candidates(2)

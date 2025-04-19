import torch
from torch import nn

from vectorrepr.modeling.encoder_decoder_wrapper import EncoderDecoderEmbedder


def test_encoder_decoder_wrapper():

    class Encoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super(Encoder, self).__init__()
            self.lstm = nn.LSTM(input_dim, latent_dim, batch_first=True)

        def forward(self, x):
            outputs, (h, c) = self.lstm(x)
            return h

    data = torch.randn(32, 10, 64)
    encoder = Encoder(64, 128)

    embedder = EncoderDecoderEmbedder(encoder)
    output = embedder(data)[-1]
    assert output.shape == (32, 128)

    decoder = nn.Linear(128, 64)
    embedder = EncoderDecoderEmbedder(encoder, decoder)
    output = embedder(data)[-1]
    assert output.shape == (32, 64)

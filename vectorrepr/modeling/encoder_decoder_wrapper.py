import torch
import torch.nn as nn


class EncoderDecoderEmbedder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module = None):
        """
        Initialize the Embedder model.

        Parameters
        ----------
            encoder : nn.Module
                The encoder network.
            decoder : nn.Module, optional
                The decoder network. If None, the Embedder will only output the encoded vector.
        """
        super(EncoderDecoderEmbedder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """前向传播函数，可以根据需要实现解码器的逻辑"""
        encoded = self.encoder(x)
        if self.decoder is not None:
            decoded = self.decoder(encoded)
            return decoded
        return encoded


# 使用示例
if __name__ == "__main__":
    # 示例：用户自定义的 Encoder 和 Decoder

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
    output = embedder(data)
    print(output.shape)

    decoder = nn.Linear(128, 64)
    embedder = EncoderDecoderEmbedder(encoder, decoder)
    output = embedder(data)
    print(output.shape)

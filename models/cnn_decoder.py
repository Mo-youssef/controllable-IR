import torch
import torch.nn as nn


class CNNDecoder(nn.Module):

    def __init__(self, output_channels: int, channel: int, latent_size: int):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, channel, 4, 2, 1), #2
            nn.ReLU(),
            nn.ConvTranspose2d(channel, channel, 4, 2, 1), #4
            nn.ReLU(),
            nn.ConvTranspose2d(channel, channel, 4, 2, 0), #10
            nn.ReLU(),
            nn.ConvTranspose2d(channel, channel, 3, 2, 0), #21
            nn.ReLU(),
            nn.ConvTranspose2d(channel, channel, 4, 2, 1), #42
            nn.ReLU(),
            nn.ConvTranspose2d(channel, output_channels, 4, 2, 1), #84
            # nn.Tanh()
        )

        print(self)

    def forward(self, features: torch.Tensor):
        return self.decoder(features.view(features.shape[0], -1, 1, 1))
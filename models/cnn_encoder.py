import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.reshape(x.size(0), -1)

class CNNEncoder(nn.Module):

    def __init__(self, input_channels: int, channels: int, latent_size: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1),
            ResBlock(channels),
            ResBlock(channels),
        )

        output_size = 21 * 21 * channels
        self.hidden_size = latent_size
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(output_size, latent_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.encoder(observation)
        return self.mlp(features)


class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)
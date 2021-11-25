import torch
import torch.nn as nn

from .cnn_encoder import CNNEncoder
from .cnn_decoder import CNNDecoder


class ControlledNetwork(nn.Module):

    def __init__(self, input_channels: int, num_actions: int,
                 hidden_size: int, channels: int, latent_size: int, encoder_out: int):
        super().__init__()
        self.num_actions = num_actions

        embedding_size = 8
        self.encoder = CNNEncoder(input_channels, channels, encoder_out)
        self.decoder = CNNDecoder(input_channels, channels, latent_size)

        self.action_embedding = nn.Embedding(num_actions, embedding_size)

        self.controllable_module = nn.Sequential(
            nn.Linear(encoder_out + embedding_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(inplace=True),
        )

        
    def forward(self, observation, action):
        action_embedding = self.action_embedding(action.long())
        hidden_observation = self.encoder(observation)
        hidden_controllable = self.controllable_module(torch.cat([hidden_observation, action_embedding], dim=-1))
        controllable_effects = self.decoder(hidden_controllable)
        return controllable_effects, hidden_controllable
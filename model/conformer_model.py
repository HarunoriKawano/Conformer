import torch
from torch import nn
from typing import Optional

from model.conformer_config import ConformerConfig
from model.conformer_encoder import ConformerEncoder
from model.conformer_subsampling import ConvSubsampling


class ConformerModel(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.subsampling_conv = ConvSubsampling(config)
        self.encoder = ConformerEncoder(config)

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        hidden_states = self.subsampling_conv(input_values)

        hidden_states = self.encoder(hidden_states, attention_mask=attention_mask)

        return hidden_states


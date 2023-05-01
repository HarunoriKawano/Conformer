import torch
from torch import nn

from model.conformer_config import ConformerConfig


class ConvSubsampling(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(1, config.hidden_size, kernel_size=3, stride=2)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2)
        self.activation2 = nn.ReLU()

    def forward(self, input_values: torch.Tensor):
        hidden_states = self.conv1(input_values.unsqueeze(1))
        hidden_states = self.activation1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.activation2(hidden_states)

        batch_size, channels, subsampled_lengths, subsampled_dim = hidden_states.size()

        hidden_states = hidden_states.permute(0, 2, 1, 3)
        outputs = hidden_states.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)

        return outputs

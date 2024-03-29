from typing import Optional
import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from model.config import Config


class SelfAttentionModule(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.self_attn = MultiHeadSelfAttentionWithRelativePosition(config)
        self.dropout = nn.Dropout(p=0.1)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            position_embeddings (torch.Tensor): with shape `(B, L, D)`
            attention_mask (torch.Tensor): with shape `(B, L)`

        Returns:
            torch.Tensor with shape`(B, L, D)`
        """
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings
        )
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class MultiHeadSelfAttentionWithRelativePosition(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=0.1)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.num_heads, self.head_size))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.num_heads, self.head_size))

        xavier_uniform_(self.linear_q.weight)
        xavier_uniform_(self.linear_k.weight)
        xavier_uniform_(self.linear_v.weight)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): with shape `(B, L, D)`
            position_embeddings (torch.Tensor): with shape `(B, L, D)`
            attention_mask (torch.Tensor): with shape `(B, L)`

        Returns:
            torch.Tensor with shape`(B, L, D)`
        """
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # `(B, L, D)` -> `(B, L, H, D/H)`
        query = self.linear_q(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)

        # `(B, L, H, D/H)` -> `(B, L, H, D/H)`
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = self._apply_relative_embeddings(
            query=query, key=key, relative_position_embeddings=position_embeddings
        )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        hidden_states = torch.matmul(probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        out = self.linear_out(hidden_states)

        return out

    def _apply_relative_embeddings(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            relative_position_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention weight with relative position by Skew algorythm.

        Args:
            query (torch.Tensor): with shape `(B, H, L, D/H)`
            key: (torch.Tensor): with shape `(B, H, L, D/H)`
            relative_position_embeddings (torch.Tensor): with shape `(L, L, D)`

        Returns:
            torch.Tensor with shape `(B, H, L, L)`

        """

        # `(L, L, D)` -> `(H, L, L, D/H)`
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(0, 1)

        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        scores_bd = (q_with_bias_v.unsqueeze(2) * proj_relative_position_embeddings.unsqueeze(0)).sum(-1)

        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores


class RelativePositionalEncoding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.positional_params = nn.Parameter(torch.randn(config.max_source_positions * 2 - 1, config.hidden_size))
        self.max_length = config.max_source_positions

    def forward(self, hidden_states: torch.Tensor):
        input_length = hidden_states.size(1)
        position_ids = torch.arange(input_length)
        relative_position_matrix = position_ids[None, :] - position_ids[:, None]
        relative_position_matrix = relative_position_matrix + self.max_length - 1

        relative_position_embeddings = self.positional_params[relative_position_matrix]

        return relative_position_embeddings


class RelativePositionalEncodingWithCLS(RelativePositionalEncoding):
    def __init__(self, config: Config):
        super().__init__(config)
        self.cls_positional_embedding = nn.Parameter(torch.randn(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        input_length = hidden_states.size(1)
        position_ids = torch.arange(input_length)
        relative_position_matrix = position_ids[None, :] - position_ids[:, None]
        relative_position_matrix = relative_position_matrix + self.max_length - 1

        relative_position_embeddings = self.positional_params[relative_position_matrix]

        relative_position_embeddings[0] = self.cls_positional_embedding
        relative_position_embeddings[:, 0] = self.cls_positional_embedding

        return relative_position_embeddings

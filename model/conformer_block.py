import torch
from torch import nn
import math
from typing import Optional

from model.conformer_config import ConformerConfig


class ConformerBlock(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()

        self.ffn1 = FeedForward(config)
        self.self_attn = SelfAttentionModule(config)
        self.conv_module = ConvolutionModule(config)
        self.ffn2 = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        residual = hidden_states
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual

        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings

        )
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.conv_module(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual

        out = self.final_layer_norm(hidden_states)

        return out


class SelfAttentionModule(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.self_attn = MultiHeadSelfAttention(config)
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings
        )
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ConformerConfig):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        batch_size, sequence_length, hidden_size = hidden_states.size()

        query = self.linear_q(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(hidden_states).view(batch_size, -1, self.num_heads, self.head_size)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = self._apply_relative_embeddings(
            query=query, key=key, relative_position_embeddings=position_embeddings
        )

        if attention_mask is not None:
            scores = scores + attention_mask

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
    ):
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.size(0), -1, self.num_heads, self.head_size
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3)

        query = query.transpose(1, 2)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)

        zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
        scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
        scores_bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, : scores_bd.size(-1) // 2 + 1]

        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores


class ConvolutionModule(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.point_wise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.glu = nn.GLU(dim=1)

        self.depth_wise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=(config.conv_depthwise_kernel_size - 1) // 2,
            groups=config.hidden_size,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.activation = nn.SiLU()

        self.point_wise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.point_wise_conv1(hidden_states)
        hidden_states = self.glu(hidden_states)

        hidden_states = self.depth_wise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.point_wise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.SiLU()
        self.intermediate_dropout = nn.Dropout(p=0.1)

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(p=0.1)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states

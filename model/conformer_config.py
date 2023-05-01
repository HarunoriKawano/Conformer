from dataclasses import dataclass


@dataclass(frozen=True)
class ConformerConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    conv_depthwise_kernel_size: int
    num_hidden_layers: int
    max_source_positions: int
    mel_filter_size: int

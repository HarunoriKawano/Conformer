from dataclasses import dataclass


@dataclass(frozen=True)
class ConformerConfig:
    hidden_size: int  # Dimension of encoder hidden states (Default: 512)
    intermediate_size: int  # Dimension of feed forward hidden states (Default: 2048)
    num_attention_heads: int  # Number of self attention heads (Default: 8)
    conv_depth_wise_kernel_size: int  # Kernel size of depth wise convolution (Default: 31)
    num_hidden_layers: int  # Number of Conformer blocks (Default 17)
    max_source_positions: int  # Maximum input length of encoder (Default 10000)
    mel_filter_size: int  # Number of mel filter banks. (Default: 80)

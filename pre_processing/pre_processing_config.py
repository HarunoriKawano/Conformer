from dataclasses import dataclass


@dataclass(frozen=True)
class PreProcessingConfig:
    mel_filter_size: int
    max_source_positions: int
    freq_mask_param: int
    time_mask_param: int
    time_mask_ratio: int

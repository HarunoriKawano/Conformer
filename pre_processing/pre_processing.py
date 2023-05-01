import torch
from torchaudio.transforms import MelSpectrogram, Resample, TimeMasking, FrequencyMasking
from torchaudio.functional import resample
from typing import Optional

from pre_processing import PreProcessingConfig


class ConformerPreProcessing:
    def __init__(self, config: PreProcessingConfig, sample_rate: int = 16000, spec_aug: bool = True):
        self.config: PreProcessingConfig = config
        self.win_length = int(config.resample_rate * config.win_time)
        self.hop_length = int(config.resample_rate * config.stride_time)
        self.resampler: Optional[Resample] = None
        if config.resample_rate != sample_rate:
            self.resampler = Resample(sample_rate, config.resample_rate)

        self.mel_sampler = MelSpectrogram(
            sample_rate=config.resample_rate,
            win_length=int(config.resample_rate * config.win_time),
            hop_length=int(config.resample_rate * config.stride_time),
            n_fft=config.n_fft,
            n_mels=config.mel_filter_size
        )

        self.spec_aug = spec_aug
        self.freq_masking = FrequencyMasking(config.freq_mask_param)
        self.time_masking = TimeMasking(config.time_mask_param, p=config.time_mask_ratio)

    def __call__(self, inputs: torch.Tensor, sample_rate=16000):
        if sample_rate != self.config.resample_rate:
            inputs = resample(inputs, sample_rate, self.config.resample_rate)
        elif self.resampler is not None:
            inputs = self.resampler(inputs)

        mel_feature = self.mel_sampler(inputs)

        if self.spec_aug:
            mel_feature = self.freq_masking(mel_feature)
            mel_feature = self.time_masking(mel_feature)

        return mel_feature

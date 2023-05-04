import torch
from torchaudio.transforms import MelSpectrogram, Resample, TimeMasking, FrequencyMasking
from torchaudio.functional import resample
from typing import Optional

from pre_processing import PreProcessingConfig


class ConformerPreProcessing:
    def __init__(self, config: PreProcessingConfig, sample_rate: int = 16000, should_spec_aug: bool = True):
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

        self.should_spec_aug = should_spec_aug
        self.freq_masking = FrequencyMasking(config.freq_mask_length)
        self.time_masking = TimeMasking(config.time_mask_length, iid_masks=True, p=config.time_mask_prop)

    def __call__(self, inputs: torch.Tensor, sample_rate=16000):
        """
        Args:
            inputs (torch.Tensor): with shape `(B, T)` or `(T)`
            sample_rate: (int): input sample rate.

        Returns:
            torch.Tensor: with shape `(B, T, D)` or `(T, D)`

        """
        if sample_rate != self.config.resample_rate:
            inputs = resample(inputs, sample_rate, self.config.resample_rate)
        elif self.resampler is not None:
            inputs = self.resampler(inputs)

        mel_feature = self.mel_sampler(inputs)

        if self.should_spec_aug:
            mel_feature = self.freq_masking(mel_feature)
            mel_feature = self.time_masking(mel_feature)

        if mel_feature.dim() == 2:
            mel_feature = mel_feature.transpose(0, 1)
        else:
            mel_feature = mel_feature.transpose(1, 2)

        return mel_feature

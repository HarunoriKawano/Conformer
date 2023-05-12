import torch
from torchaudio.transforms import MelSpectrogram, Resample, TimeMasking, FrequencyMasking
from torchaudio.functional import resample
from typing import Optional

from pre_processing import Config


class ConformerPreProcessing:
    def __init__(
        self,
        config: Config,
        sample_rate: int = 16000,
        should_spec_aug: bool = True,
        noise_scale: float = 1e-4
    ):
        self.config: Config = config
        self.noise_scale = noise_scale
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

    def __call__(self, inputs: torch.Tensor, sample_rate=16000) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): with shape `(T)` or `(B, T)`
            sample_rate (int): input sampling rate.

        Returns:
            torch.Tensor with shape `(T, D)` or `(B, T, D)`

        """
        if sample_rate != self.config.resample_rate:
            inputs = resample(inputs, sample_rate, self.config.resample_rate)
        elif self.resampler is not None:
            inputs = self.resampler(inputs)

        # Add noise for log scaling
        noise = torch.randn(inputs.size(), device=inputs.device) * self.noise_scale
        inputs += noise

        mel_feature = self.mel_sampler(inputs)
        log_mel_feature = mel_feature.log2()

        if self.should_spec_aug:
            log_mel_feature = self.freq_masking(log_mel_feature)
            log_mel_feature = self.time_masking(log_mel_feature)

        if log_mel_feature.dim() == 2:
            log_mel_feature = log_mel_feature.transpose(0, 1)
        else:
            log_mel_feature = log_mel_feature.transpose(1, 2)

        return log_mel_feature

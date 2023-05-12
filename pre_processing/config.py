from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    mel_filter_size: int  # Number of mel filter banks. (Default: 80)
    freq_mask_length: int  # Maximum possible length of freq mask. (Default: 27)
    time_mask_length: int  # Maximum possible length of time mask. (Default: 10)
    time_mask_prop: int  # Maximum proportion of time steps. (Default: 0.05)
    resample_rate: int  # Sampling rate of input values. (Default: 16000)
    win_time: float  # Window size (sec). (Default: 0.025)
    stride_time: float  # Length of hop between STFT windows (sec). (Default: 0.01)
    n_fft: int  # Size of FFT. (Default: 2048)

import os
import json

import torch
from torchaudio.datasets import LibriLightLimited
import matplotlib.pyplot as plt

from config import Config
from model import ConformerModel
from pre_processing import ConformerPreProcessing


def plot_mel_spectrogram(wave, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(wave.transpose(0 ,1), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


if __name__ == '__main__':
    test_data_path = "data"
    os.makedirs(test_data_path, exist_ok=True)
    test_dataset = LibriLightLimited(test_data_path, subset="10min", download=True)

    with open("config/middle_config.json", "r", encoding="utf-8") as f:
        config = Config(**json.load(f))
    pre_processor = ConformerPreProcessing(config)
    model = ConformerModel(config)

    test_data = test_dataset[0]
    wav = test_data[0]
    sr = test_data[1]

    # wave to log_mel_spectrogram
    log_mel_spectrogram = pre_processor(wav, sr)
    plot_mel_spectrogram(log_mel_spectrogram[0])
    input_lengths = torch.tensor([log_mel_spectrogram.size(1)], dtype=torch.long)

    out, input_lengths = model(log_mel_spectrogram, input_lengths)
    print(out.shape)  # `(B, L, D)`
    print(input_lengths)  # `(B)`

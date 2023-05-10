import os
import json

import torch
from torchaudio.datasets import LibriLightLimited
from torch import nn
from torch.nn.functional import log_softmax
import matplotlib.pyplot as plt

from config import Config
from model import ConformerModel
from pre_processing import ConformerPreProcessing


def plot_mel_spectrogram(wave, title=None, y_label="freq_bin", aspect="auto", x_max=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(y_label)
    axs.set_xlabel("frame")
    im = axs.imshow(wave.transpose(0, 1), origin="lower", aspect=aspect)
    if x_max:
        axs.set_xlim((0, x_max))
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

    num_phonemes = 45
    targets = torch.randint(1, num_phonemes, size=(1, 100))  # 0: blank
    target_lengths = torch.tensor([100, ])

    out_linear = nn.Linear(out.size(-1), num_phonemes + 1)
    loss_func = nn.CTCLoss()

    # `(B, L, D)` -> `(L, B, N)`
    probs = log_softmax(out_linear(out).transpose(0, 1), dim=-1)

    loss = loss_func(probs, targets, input_lengths, target_lengths)
    print(loss)
    loss.backward()

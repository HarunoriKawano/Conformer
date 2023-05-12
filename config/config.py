from dataclasses import dataclass

from model import Config as Wav2vec2Config
from pre_processing import Config as PreProcessingConfig


@dataclass(frozen=True)
class Config(Wav2vec2Config, PreProcessingConfig):
    pass

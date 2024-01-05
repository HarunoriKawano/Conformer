from dataclasses import dataclass

from model import Config as ConformerConfig
from pre_processing import Config as PreProcessingConfig


@dataclass(frozen=True)
class Config(ConformerConfig, PreProcessingConfig):
    pass

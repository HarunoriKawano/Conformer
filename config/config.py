from dataclasses import dataclass

from model import ConformerConfig
from pre_processing import PreProcessingConfig


@dataclass(frozen=True)
class Config(ConformerConfig, PreProcessingConfig):
    pass

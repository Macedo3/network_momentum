"""Network Momentum para equities globais (arXiv:2308.11294 adaptado)."""

from .config import AppConfig, load_config
from .pipeline import PipelineOptions, run_full_pipeline

__all__ = ["AppConfig", "load_config", "PipelineOptions", "run_full_pipeline"]
__version__ = "1.0.0"

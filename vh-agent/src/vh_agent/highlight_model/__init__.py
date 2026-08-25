"""Trainable narrative-transition highlight localization."""

from .config import HighlightModelConfig
from .trainer import fit_highlight_model

__all__ = ["HighlightModelConfig", "fit_highlight_model"]

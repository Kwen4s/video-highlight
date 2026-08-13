"""Event-aware short-drama highlight detection."""

from .models import (
    DetectionResult,
    DetectionTask,
)
from .pipeline import HighlightDetectionService

__all__ = [
    "DetectionResult",
    "DetectionTask",
    "HighlightDetectionService",
]
__version__ = "0.13.0"

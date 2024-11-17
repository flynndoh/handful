"""Shared types and protocols."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol, Tuple
import numpy as np


class Color(Enum):
    """Standard colors for visualization."""
    WHITE = (224, 224, 224)
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    GREEN = (0, 128, 0)
    BLUE = (255, 0, 0)


@dataclass
class HandLandmarks:
    """Stores processed hand landmark data."""
    fingers_up: List[bool]
    num_fingers_up: int
    landmark_points: List[Tuple[int, int]]


@dataclass
class ProcessedFrame:
    """Container for a processed frame and its analysis results."""
    frame: np.ndarray
    hand_data: Optional[List[HandLandmarks]]
    timestamp: float


class FrameSource(Protocol):
    """Protocol for frame sources (cameras, streams, etc.)"""

    def start(self) -> None:
        """Start the frame source"""
        ...

    def stop(self) -> None:
        """Stop the frame source"""
        ...

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the next frame"""
        ...
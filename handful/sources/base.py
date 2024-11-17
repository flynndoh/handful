"""Base classes for frame sources."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from handful.core.types import FrameSource


class BaseFrameSource(ABC, FrameSource):
    """Abstract base class for frame sources."""

    @abstractmethod
    def start(self) -> None:
        """Start the frame source."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the frame source."""
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the next frame."""
        pass

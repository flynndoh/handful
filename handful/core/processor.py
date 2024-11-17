from typing import Generator, Callable, Optional

import cv2
import numpy as np

from handful.core.tracker import HandTracker
from handful.core.types import FrameSource, ProcessedFrame


class StreamProcessor:
    """Handles frame processing pipeline for video streams"""

    def __init__(
        self,
        frame_source: FrameSource,
        tracker: Optional[HandTracker] = None,
        preprocessing_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        postprocessing_fn: Optional[Callable[[ProcessedFrame], np.ndarray]] = None
    ):
        """Initialize the stream processor.
        :param frame_source: Source of video frames (must implement FrameSource protocol)
        :param tracker: HandTracker instance (creates new one if None)
        :param preprocessing_fn: Optional function to preprocess frames before tracking
        :param postprocessing_fn: Optional function to postprocess frames after tracking
        """
        self.frame_source = frame_source
        self.tracker = tracker or HandTracker()
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn
        self._running = False

    def process_frames(self) -> Generator[ProcessedFrame, None, None]:
        """Process frames from the source and yield results.

        Yields:
            ProcessedFrame objects containing the frame and analysis results
        """
        self.frame_source.start()
        self._running = True

        try:
            while self._running:
                # Get frame from source
                frame = self.frame_source.get_frame()
                if frame is None:
                    continue

                # Apply preprocessing if specified
                if self.preprocessing_fn:
                    frame = self.preprocessing_fn(frame)

                # Process frame with hand tracker
                processed_frame, hand_data = self.tracker.process_frame(frame)

                # Create processed frame object
                result = ProcessedFrame(
                    frame=processed_frame,
                    hand_data=hand_data,
                    timestamp=cv2.getTickCount() / cv2.getTickFrequency()
                )

                # Apply postprocessing if specified
                if self.postprocessing_fn:
                    result.frame = self.postprocessing_fn(result)

                yield result

        finally:
            self.stop()

    def stop(self):
        """Stop processing frames"""
        self._running = False
        self.frame_source.stop()
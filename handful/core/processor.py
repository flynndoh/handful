from typing import Callable, Optional
from queue import Queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from handful.core.tracker import HandTracker
from handful.core.types import FrameSource, ProcessedFrame


@dataclass
class ProcessingSubscriber:
    """Represents a subscriber to the processing pipeline"""
    name: str
    queue: Queue
    max_queue_size: int = 10

    def put(self, data: ProcessedFrame) -> None:
        """Put data into subscriber queue, dropping oldest if full"""
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except:
                pass
        self.queue.put(data)


class StreamProcessor:
    """Handles frame processing pipeline for video streams with multiple subscribers"""

    def __init__(
            self,
            frame_source: FrameSource,
            tracker: Optional[HandTracker] = None,
            preprocessing_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            postprocessing_fn: Optional[Callable[[ProcessedFrame], np.ndarray]] = None
    ):
        self.frame_source = frame_source
        self.tracker = tracker or HandTracker()
        self.preprocessing_fn = preprocessing_fn
        self.postprocessing_fn = postprocessing_fn
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._subscribers: dict[str, ProcessingSubscriber] = {}
        self._lock = threading.Lock()

    def subscribe(self, name: str, max_queue_size: int = 10) -> Queue:
        """Add a new subscriber to receive processed frames

        Returns:
            Queue that will receive ProcessedFrame objects
        """
        with self._lock:
            if name in self._subscribers:
                return self._subscribers[name].queue

            queue = Queue(maxsize=max_queue_size)
            self._subscribers[name] = ProcessingSubscriber(name, queue, max_queue_size)
            return queue

    def unsubscribe(self, name: str) -> None:
        """Remove a subscriber"""
        with self._lock:
            if name in self._subscribers:
                del self._subscribers[name]

    def _process_frames(self):
        """Internal frame processing loop"""
        self.frame_source.start()

        try:
            while self._running:
                # Get frame from source
                frame = self.frame_source.get_frame()
                if frame is None:
                    if self._running:  # Only sleep if we're still supposed to be running
                        time.sleep(0.01)  # Short sleep to prevent busy waiting
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

                # Distribute to all subscribers
                with self._lock:
                    for subscriber in self._subscribers.values():
                        subscriber.put(result)

        finally:
            self.frame_source.stop()

    def start(self):
        """Start processing frames"""
        if self._running:
            return

        self._running = True
        self._processing_thread = threading.Thread(
            target=self._process_frames,
            daemon=True
        )
        self._processing_thread.start()

    def stop(self):
        """Stop processing frames"""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)
            self._processing_thread = None



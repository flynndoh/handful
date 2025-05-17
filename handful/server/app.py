import time
from queue import Queue, Empty
from typing import Generator, Optional
import logging
from threading import Event

import cv2
from flask import Flask, Response, render_template

from handful.core.processor import StreamProcessor
from handful.core.types import ProcessedFrame

logger = logging.getLogger(__name__)


class StreamServer:
    """Handles serving processed video frames over HTTP"""

    def __init__(
            self,
            processor: StreamProcessor,
            host: str = "0.0.0.0",
            port: int = 5000,
    ):
        self.processor = processor
        self.host = host
        self.port = port
        self.frame_queue: Optional[Queue] = None
        self._running = False
        self._current_fps = 0
        self._last_frame_time = 0
        self._frames_processed = 0
        self._shutdown_event = Event()

        # Initialize the queue immediately in constructor
        logger.info("Subscribing to processor")
        self.frame_queue = self.processor.subscribe('web_server')

        self.app = self._create_app()

    def _create_app(self) -> Flask:
        """Create and configure Flask application"""
        app = Flask(
            __name__,
            template_folder='templates',
            static_folder='static'
        )

        @app.route('/')
        def index():
            return render_template("index.html")

        @app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @app.route('/stats')
        def stats():
            return {
                'queue_size': self.frame_queue.qsize() if self.frame_queue else 0,
                'is_running': self._running,
                'fps': self._current_fps
            }

        return app

    def _generate_frames(self) -> Generator[bytes, None, None]:
        """Generate MJPEG stream from processed frames"""
        if not self.frame_queue:
            logger.error("Frame queue not initialized!")
            return

        while self._running and not self._shutdown_event.is_set():
            try:
                processed: ProcessedFrame = self.frame_queue.get(timeout=0.1)
                if processed and processed.frame is not None:
                    # Update FPS calculation
                    current_time = time.time()
                    if current_time - self._last_frame_time >= 1.0:
                        self._current_fps = self._frames_processed
                        self._frames_processed = 0
                        self._last_frame_time = current_time
                    self._frames_processed += 1

                    # Encode and yield the frame
                    ret, buffer = cv2.imencode('.jpg', processed.frame)
                    if ret:
                        frame_data = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' +
                               frame_data +
                               b'\r\n')
            except Empty:
                continue  # Just try again if queue is empty
            except Exception as e:
                if not self._shutdown_event.is_set():  # Only log if not shutting down
                    logger.debug(f"Frame generation error: {str(e)}")
                time.sleep(0.01)  # Prevent busy waiting
                continue

    def start(self):
        """Start the stream server"""
        if self._running:
            logger.warning("Server is already running")
            return

        self._running = True
        self._current_fps = 0
        self._last_frame_time = time.time()
        self._frames_processed = 0
        self._shutdown_event.clear()

        # Start Flask server (this will block until server stops)
        logger.info(f"Starting Flask server on {self.host}:{self.port}")
        try:
            self.app.run(
                host=self.host,
                port=self.port,
                threaded=True,
                use_reloader=False
            )
        finally:
            self.stop()

    def stop(self):
        """Stop the stream server and cleanup resources"""
        if not self._running:
            return

        logger.info("Stopping stream server...")
        self._running = False
        self._shutdown_event.set()

        if self.processor and self.frame_queue:
            try:
                self.processor.unsubscribe('web_server')
                self.frame_queue = None
            except Exception as e:
                logger.error(f"Error unsubscribing from processor: {e}")

        logger.info("Stream server stopped")

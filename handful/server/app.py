import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Generator, Optional

import cv2
from flask import Flask, Response, render_template

from handful.core.processor import StreamProcessor
from handful.core.types import ProcessedFrame


@dataclass
class StreamServer:
    """Handles serving processed video frames over HTTP"""

    def __init__(
        self,
        processor: StreamProcessor,
        host: str = "0.0.0.0",
        port: int = 5000,
        frame_queue_size: int = 10
    ):
        """Initialize the stream server.
        :param processor: Stream processor instance
        :param host: Host address to bind to
        :param port: Port to listen on
        :param frame_queue_size: Maximum number of frames to queue
        """
        self.processor = processor
        self.host = host
        self.port = port
        self.frame_queue = Queue(maxsize=frame_queue_size)
        self.app = self._create_app()
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None

    def _create_app(self) -> Flask:
        """Create and configure Flask application"""
        app = Flask(
            __name__,
            template_folder='templates',
            static_folder='static'
        )

        @app.route('/')
        def index():
            """Serve the main page."""
            return render_template("index.html")

        @app.route('/video_feed')
        def video_feed():
            """Stream the processed video frames."""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @app.route('/stats')
        def stats():
            """Return current processing statistics."""
            return {
                'queue_size': self.frame_queue.qsize(),
                'is_running': self._running,
                'fps': self._current_fps
            }

        return app


    def _generate_frames(self) -> Generator[bytes, None, None]:
        """Generate MJPEG stream from processed frames.

        Yields:
            JPEG-encoded frame data with MIME multipart headers
        """
        while self._running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() +
                           b'\r\n')
            except:
                continue  # No frame available, try again


    def _process_frames(self):
        """Process frames in a separate thread."""
        last_frame_time = time.time()
        frames_processed = 0

        def handle_frame(processed: ProcessedFrame):
            nonlocal last_frame_time, frames_processed

            # Update FPS calculation
            current_time = time.time()
            frames_processed += 1
            if current_time - last_frame_time >= 1.0:
                self._current_fps = frames_processed
                frames_processed = 0
                last_frame_time = current_time

            # Add frame to queue, dropping oldest if full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            self.frame_queue.put(processed.frame)

        # Start processing frames
        for processed_frame in self.processor.process_frames():
            if not self._running:
                break
            handle_frame(processed_frame)


    def start(self):
        """Start the stream server."""
        self._running = True
        self._current_fps = 0

        # Start frame processing in a separate thread
        self._processing_thread = threading.Thread(
            target=self._process_frames,
            daemon=True
        )
        self._processing_thread.start()

        # Start Flask server
        self.app.run(
            host=self.host,
            port=self.port,
            threaded=True,
            use_reloader=False  # Disable reloader to prevent double startup
        )


    def stop(self):
        """Stop the stream server."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
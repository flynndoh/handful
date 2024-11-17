import argparse
import cv2

from handful.core.processor import StreamProcessor
from handful.core.tracker import HandTracker
from handful.server.app import StreamServer
from handful.sources.mjpeg import MJPEGStreamClient


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run hand tracking stream processor with optional configuration.")
    parser.add_argument(
        "--stream_url",
        type=str,
        required=True,
        help="URL of the MJPEG stream (e.g., 'http://192.168.0.117:8080/stream')."
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=None,
        help="Width to resize the frame for preprocessing (optional)."
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=None,
        help="Height to resize the frame for preprocessing (optional)."
    )
    parser.add_argument(
        "--debug_visualization",
        action="store_true",
        default=True,
        help="Enable debug visualization for tracked hands."
    )
    parser.add_argument(
        "--restream_port",
        type=int,
        default=5000,
        help="Port to use to re-stream the input video with debug visualizations (if enabled)"
    )
    args = parser.parse_args()

    # Create source and tracker
    source = MJPEGStreamClient(args.stream_url)
    tracker = HandTracker()

    # Define optional preprocessing and postprocessing functions
    preprocessing_fn = None
    if args.resize_width and args.resize_height:
        preprocessing_fn = lambda frame: cv2.resize(frame, (args.resize_width, args.resize_height))

    postprocessing_fn = None
    if args.debug_visualization:
        postprocessing_fn = lambda proc: tracker.create_debug_visualization(
            proc.frame,
            proc.hand_data
        )

    # Create processor with optional pre/post-processing
    processor = StreamProcessor(
        frame_source=source,
        tracker=tracker,
        preprocessing_fn=preprocessing_fn,
        postprocessing_fn=postprocessing_fn
    )

    # Create and start server
    server = StreamServer(processor, port=args.restream_port)
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()

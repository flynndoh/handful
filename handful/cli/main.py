import time

import click
from typing import Optional, List
import logging
from pathlib import Path
import threading
import signal
from dataclasses import dataclass, field

from handful.core.processor import StreamProcessor
from handful.core.tracker import HandTracker
from handful.server.app import StreamServer
from handful.sources.mjpeg import MJPEGStreamClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Holds CLI configuration and pipeline state"""
    debug: bool = False
    config_file: Optional[Path] = None
    processor: Optional['StreamProcessor'] = None
    running_threads: List[threading.Thread] = field(default_factory=list)
    shutdown_event: threading.Event = field(default_factory=threading.Event)
    server: Optional[StreamServer] = None  # Keep reference to server instance

    def add_thread(self, thread: threading.Thread):
        """Add a background thread to be joined later"""
        self.running_threads.append(thread)

    def cleanup(self):
        """Signal threads to stop and wait for completion"""
        self.shutdown_event.set()

        # Cleanup server if it exists
        if self.server:
            self.server.stop()

        # Cleanup processor if it exists
        if self.processor:
            self.processor.stop()

        for thread in self.running_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)


@click.group(chain=True, invoke_without_command=True)
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
@click.option('--config', type=click.Path(path_type=Path), help='Configuration file path')
@click.pass_context
def cli(ctx: click.Context, debug: bool, config: Optional[Path]):
    """Pipeline CLI that supports parallel command execution"""
    ctx.ensure_object(Config)
    config_obj = ctx.obj
    config_obj.debug = debug
    config_obj.config_file = config

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, cleaning up...")
        config_obj.cleanup()
        ctx.exit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


@cli.command()
@click.option('--url', required=True, help='MJPEG stream URL')
@click.pass_context
def mjpeg(ctx: click.Context, url: str):
    """Set up an MJPEG source"""
    config: Config = ctx.obj

    config.source_client = MJPEGStreamClient(url)
    config.tracker = HandTracker()
    config.processor = StreamProcessor(config.source_client, config.tracker)

    # Start the processor
    config.processor.start()
    logger.info(f"Source configured with URL: {url}")


@cli.command()
@click.pass_context
def send_hand_data(ctx: click.Context):
    """Send the thumb position to the servo control webapp"""
    config: Config = ctx.obj

    if not config.processor:
        raise click.UsageError("You must set up a source before running this.")

    queue = config.processor.subscribe('send_hand_data')

    def run():
        try:
            while not config.shutdown_event.is_set():
                try:
                    processed = queue.get(timeout=0.1)
                    if processed.hand_data:
                        logger.info(processed.hand_data)
                except:
                    continue
        except Exception as e:
            logger.error(f"Error in hand data thread: {e}")
        finally:
            config.processor.unsubscribe('send_hand_data')
            logger.info("Hand data processing stopped")

    thread = threading.Thread(target=run, daemon=True)
    config.add_thread(thread)
    thread.start()
    logger.info("Hand data processing started in background")


@cli.command()
@click.option('--port', default=5000, help='Web server port')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.pass_context
def serve(ctx: click.Context, port: int, host: str):
    """Start the web server"""
    config: Config = ctx.obj

    if not config.processor:
        raise click.UsageError("You must set up a source before starting the server.")

    logger.info(f"Starting web server on {host}:{port}")

    # Create server instance
    server = StreamServer(config.processor, host=host, port=port)
    config.server = server  # Store reference for cleanup

    def run_server():
        try:
            server.start()
        except Exception as e:
            if not config.shutdown_event.is_set():  # Only log if not shutting down
                logger.error(f"Error in server thread: {e}")
        finally:
            logger.info("Web server stopped")

    # Start server in background thread
    thread = threading.Thread(target=run_server, daemon=True)
    config.add_thread(thread)
    thread.start()

    # Give the server a moment to initialize
    time.sleep(0.5)

    logger.info("Web server started in background")

@cli.result_callback()
def process_pipeline(processors, **kwargs):
    """Keep the pipeline running until interrupted"""
    if processors:  # Only process if commands were run
        config = click.get_current_context().obj
        try:
            # Block main thread with event wait
            logger.info("Main thread now waiting until exit is requested.")
            config.shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            config.cleanup()


if __name__ == '__main__':
    cli()

import click
from typing import Optional
import logging
from pathlib import Path

import cv2
import yaml

from handful.core.processor import StreamProcessor
from handful.core.tracker import HandTracker
from handful.server.app import StreamServer
from handful.sources.mjpeg import MJPEGStreamClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Holds CLI configuration"""

    def __init__(self):
        self.debug: bool = False
        self.config_file: Optional[Path] = None


pass_config = click.make_pass_decorator(Config, ensure=True)


def load_config(ctx: click.Context, config_file: Optional[Path]) -> dict:
    """Load configuration from file if provided"""
    if config_file and config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    return {}


@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
@click.option('--config', type=click.Path(path_type=Path), help='Configuration file path')
@click.pass_context
def cli(ctx: click.Context, debug: bool, config: Optional[Path]):
    """Hand tracking system CLI"""
    ctx.obj = Config()
    ctx.obj.debug = debug
    ctx.obj.config_file = config

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.debug(f"Debug mode: {debug}")
    logger.debug(f"Config file: {config}")


@cli.group()
def source():
    """Manage frame sources"""
    pass


@source.command()
@click.option('--url', required=True, help='MJPEG stream URL')
@click.option('--display/--no-display', default=False, help='Show preview window')
@click.option('--server/--no-server', default=True, help='Start web server')
@click.option('--port', default=5000, help='Web server port')
@pass_config
def mjpeg(config: Config, url: str, display: bool, server: bool, port: int):
    """Use MJPEG stream as frame source"""
    cfg = load_config(click.get_current_context(), config.config_file)

    source_client = MJPEGStreamClient(url)
    tracker = HandTracker()
    processor = StreamProcessor(source_client, tracker)

    if server:
        logger.info(f"Starting web server on port {port}")
        server = StreamServer(processor, port=port)
        server.start()
    elif display:
        logger.info("Starting display mode")
        try:
            for processed in processor.process_frames():
                cv2.imshow("Hand Tracking", processed.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()


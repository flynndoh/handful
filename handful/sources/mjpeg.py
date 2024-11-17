import cv2
import numpy as np
import requests
from threading import Thread

from handful.sources.base import BaseFrameSource


class MJPEGStreamClient(BaseFrameSource):

    def __init__(self, url: str, boundary: str ="mjpegstream"):
        """
        Initializes the MJPEG stream client.
        :param url: The URL of the MJPEG stream.
        :param boundary: The boundary string used to separate frames in the stream.
        """
        self.url = url
        self.boundary = boundary.encode()  # Ensure the boundary is in bytes
        self.running = False
        self.frame = None  # Holds the latest frame
        self.thread = None

    def start(self):
        """
        Starts the video stream consumption in a separate thread.
        """
        self.running = True
        self.thread = Thread(target=self._consume_stream, daemon=True)
        self.thread.start()

    def stop(self):
        """
        Stops the video stream consumption.
        """
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def _consume_stream(self):
        """
        Consumes the MJPEG stream and decodes frames in real-time.
        """
        with requests.get(self.url, stream=True) as response:
            if response.status_code != 200:
                print(f"Failed to connect to stream: {response.status_code}")
                self.running = False
                return

            buffer = b""  # Buffer to hold data between boundaries
            for chunk in response.iter_content(chunk_size=4096):
                if not self.running:
                    break

                buffer += chunk
                while True:
                    # Find the start of the next frame
                    start = buffer.find(b"--" + self.boundary)
                    if start == -1:
                        break

                    # Find the end of the frame
                    end = buffer.find(b"--" + self.boundary, start + len(self.boundary) + 2)
                    if end == -1:
                        break

                    # Extract frame data (with headers)
                    frame_block = buffer[start + len(self.boundary) + 2: end]
                    buffer = buffer[end:]  # Update the buffer

                    # Strip headers from the frame data
                    header_end = frame_block.find(b"\n\r\n")
                    if header_end != -1:
                        frame_data = frame_block[header_end + 4:-2]  # Skip the header and trailing /r/n

                        # HACK: Try to correct the byte ordering to convince opencv to read it
                        frame_data = bytearray(frame_data)
                        open_cv_jpeg_soi = b'\xff\xd8\xff\x00'
                        if frame_data[:4] == b'\xd8\xff\xe0\x00':
                            frame_data[:4] = open_cv_jpeg_soi

                        if frame_data.startswith(open_cv_jpeg_soi):
                            try:
                                frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                                if frame is not None:
                                    self.frame = frame
                            except Exception as e:
                                print(f"Frame decoding error: {e}")
                        else:
                            print("Invalid frame data detected. Skipping.")

    def get_frame(self):
        """
        Retrieves the latest frame from the stream.
        @return: numpy.ndarray: The latest frame, or None if no frame is available.
        """
        return self.frame
